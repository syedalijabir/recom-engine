#!/usr/bin/env python3
"""
HMM-based movie recommendations.

This module trains a global Hidden Markov Model (HMM) on user rating sequences
and recommends movies by predicting the distribution of the next (genre, rating_bin)
observation. It separates responsibilities into focused classes:

- DataLoader:            Reads raw CSV inputs.
- Vocabulary:            Maps genres/ratings to discrete tokens (and back).
- SequenceBuilder:       Builds sorted, tokenized user sequences.
- HoldoutSplitter:       Per-user last-K-movies validation split.
- HMMBackend:            Wraps hmmlearn models (Categorical/Multinomial) behind one API.
- CandidateScorer:       Aggregates genre mass to movie-level scores.
- Recommender:           User-facing topN/top1 recommendations.
- Evaluator:             Likelihood + recall@K/hit-rate + accuracy-vs-history.
- Pipeline:              Orchestrates end-to-end training, evaluation, and outputs.

Inputs (current directory):
  - hmm_training_ratings.csv  # userId,movieId,rating,timestamp
  - movies_long.csv           # movieId,clean_title,year,genre

Outputs (out_hmm/):
  - observations_1.npz
  - hmm_model.npz
  - accuracy_by_movies_1.csv
  - accuracy_by_rows_1.csv
  - eval_top1_last_movie_1.csv
  - user_topN_<userId>.csv

Notes:
  - Observations are discrete tokens from (genre_id, rating_bin).
  - A single global HMM is trained across all users; sequences are separated via 'lengths'.
"""


from __future__ import annotations

import dataclasses
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import CategoricalHMM #backends: prefer CategoricalHMM; fallback to MultinomialHMM
import time



## Configuration
@dataclass
class HMMConfig:
    """Configuration for the HMM recommender.

        Attributes:
            n_states: Number of latent HMM states.
            rating_bins: Two cut points for 3 rating bins (low, mid, high).
            top_n: Number of recommendations to produce for a user.
            outdir: Output directory for artifacts/CSVs.
            random_state: Seed used by hmmlearn.
            verbose_hmm: If True, hmmlearn prints per-iteration logs.
            holdout_last_k_movies: Size of per-user validation holdout (last K distinct movies).
            ratings_path: Path to ratings CSV.
            movies_long_path: Path to movies metadata CSV.
        """
    n_states: int = 6
    rating_bins: Tuple[float, ...] = (2.5, 3.5)
    top_n: int = 20
    outdir: Path = Path("out_hmm")
    random_state: int = 0
    verbose_hmm: bool = True
    holdout_last_k_movies: int = 100
    alpha_pop_backoff: float = 0.10  # blend to global popularity at token level
    beta_item_pop: float = 0.20  # blend to movie popularity at ranking time
    gamma_user_genre: float = 0.20  # blend user genre history into next-genre mass
    delta_year: float = 0.0         # blend user release-year preference into ranking
    min_candidate_pop: int = 5  # filter super-rare candidates (train count)
    # IO paths (can be overridden)
    ratings_path: Path = Path("data/hmm_training_ratings.csv")
    movies_long_path: Path = Path("movies_long.csv")

## IO

class DataLoader:
    """Reads raw ratings and movies metadata from CSVs."""
    def __init__(self, ratings_path: Path, movies_long_path: Path):
        self.ratings_path = ratings_path
        self.movies_long_path = movies_long_path

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load ratings and movies tables.
        Returns: (ratings, movies_long) dataframes.  Ratings include a UTC 'datetime' column."""
        rp, mp = self.ratings_path, self.movies_long_path
        if not rp.exists():
            raise FileNotFoundError(f"Missing ratings at {rp}") # If either CSV is missing
        if not mp.exists():
            raise FileNotFoundError(f"Missing movies at {mp}") #If either CSV is missing

        r = pd.read_csv(
            rp,
            dtype={"userId": "int64", "movieId": "int64", "rating": "float64", "timestamp": "int64"},
        )
        r["datetime"] = pd.to_datetime(r["timestamp"], unit="s", utc=True)

        mlong = pd.read_csv(
            mp,
            dtype={"movieId": "int64", "clean_title": "string", "year": "Int64", "genre": "string"},
        )
        return r, mlong

## Vocab/Tokens
class Vocabulary:
    """Maps (genre, rating_bin) pairs to discrete observation tokens.

    The vocabulary defines:
      - genre_to_id / id_to_genre: integer mapping for genres
      - rating_bin: integer bin index in 0..R-1, where R = len(rating_bins)+1
      - token: genre_id * R + rating_bin
      - token_to_tuple: reverse map from token -> (genre_id, rating_bin)
    """
    def __init__(self, rating_bins: Tuple[float, float]):
        self.rating_bins = tuple(sorted(rating_bins))
        self.genre_to_id: Dict[str, int] = {}
        self.id_to_genre: Optional[np.ndarray] = None
        self.token_to_tuple: Optional[np.ndarray] = None
        self.M: Optional[int] = None

    def _bin_rating(self, r: float) -> int:
        """
        Map a numeric rating to a bin index 0..R-1, where R = len(rating_bins) + 1.

        If rating_bins = (b0, b1, ..., b_{B-1}), bins are:
          0: r <= b0
          1: b0 < r <= b1
          ...
          B-1: b_{B-2} < r <= b_{B-1}
          B: b_{B-1} < r
        """
        edges = self.rating_bins
        B = len(edges)

        if pd.isna(r):
            # fall back to "middle" bin if rating is missing
            return B // 2

        for i, thr in enumerate(edges):
            if r <= thr:
                return i
        return B  # above last threshold → last bin

    def build_from_dataframe(self, rx: pd.DataFrame) -> None:
        """Initialize the genre vocabulary from a dataframe.
            Args: rx: Dataframe containing a 'genre' column.
            Side Effects:
                Sets genre_to_id, id_to_genre, token_to_tuple, and M (vocab size)."""
        genres = sorted(rx["genre"].astype(str).unique().tolist())
        self.genre_to_id = {g: i for i, g in enumerate(genres)}
        self.id_to_genre = np.array(genres, dtype=object)

        # R discrete rating bins = len(edges) + 1
        R = len(self.rating_bins) + 1
        G = len(genres)
        self.M = G * R
        self.token_to_tuple = np.array(
            [(g, rb) for g in range(G) for rb in range(R)],
            dtype=np.int32
        )

    def encode_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rating_bin/genre_id/token columns to a dataframe.
        Args: df: Dataframe with 'rating' and 'genre' columns.
        Returns: A copy of df with ['rating_bin', 'genre_id', 'token'] added.
        """
        if not self.genre_to_id:
            raise RuntimeError("Call build_from_dataframe() before encode_rows().") # If build_from_dataframe() has not been called.

        df = df.copy()
        df["genre"] = df["genre"].fillna("(no genres listed)").astype(str)
        df["rating_bin"] = df["rating"].map(self._bin_rating).astype("int64")
        df["genre_id"] = df["genre"].map(self.genre_to_id).astype("int64")
        # Note: R = len(self.rating_bins) + 1
        R = len(self.rating_bins) + 1
        df["token"] = (df["genre_id"] * R + df["rating_bin"]).astype("int64")
        return df

## Sequences
class SequenceBuilder:
    """Builds sorted, tokenized user sequences ready for HMM training."""
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def build(self, ratings: pd.DataFrame, movies_long: pd.DataFrame) -> pd.DataFrame:
        """Merge ratings with movies, sort rows, and encode tokens.
            Args:
                ratings: userId, movieId, rating, timestamp/datetime...
                movies_long: movie metadata with 'genre'.
            Returns: Tokenized, chronologically sorted dataframe."""
        rx = ratings.merge(movies_long, on="movieId", how="left", validate="many_to_many")
        rx = rx.sort_values(["userId", "datetime", "movieId", "genre"], kind="mergesort").reset_index(drop=True)
        # initialize vocab from rx genres, then encode rows
        self.vocab.build_from_dataframe(rx)
        rx = self.vocab.encode_rows(rx)
        return rx

    @staticmethod
    def to_X_lengths(rx: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
        """Convert a tokenized dataframe to hmmlearn (X, lengths) format.
            Args: rx: Tokenized dataframe with a 'token' column.
            Returns:
                X: ndarray of shape (N, 1) with integer tokens.
                lengths: list of per-user sequence lengths (in rows)."""
        X = rx["token"].to_numpy(dtype=np.int32).reshape(-1, 1)
        lengths = rx.groupby("userId")["token"].size().astype(int).tolist()
        return X, lengths

    @staticmethod
    def tokens_to_onehot(tokens: np.ndarray, M: int) -> np.ndarray:
        """One-hot encode integer tokens for MultinomialHMM.
            Args:
                tokens: Integer tokens of shape (N,) or (N, 1).
                M: Vocabulary size (# of unique tokens).
            Returns: ndarray of shape (N, M) with one-hot rows."""
        t = tokens.reshape(-1)
        X = np.zeros((t.size, M), dtype=np.int64)
        X[np.arange(t.size), t] = 1
        return X

## Split data
class HoldoutSplitter:
    """Per-user last-K-distinct-movies validation split."""
    def __init__(self, last_k_movies: int):
        self.k = int(last_k_movies)

    def split(self, rx_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split into train/valid by holding out each user's last K distinct movies.
            The "last" ordering uses each movie's most recent rating timestamp.
            Args: rx_all: Tokenized, chronologically sortable dataframe.
            Returns: (rx_train, rx_valid) dataframes.
            Raises:
                    SystemExit: If no users satisfy the (K+1) distinct movies requirement,
                                or if after holdout no user has training data.
                """
        rx = rx_all.copy()
        # Require at least (k+1) distinct movies per user to leave train+valid
        counts = rx.groupby("userId")["movieId"].nunique()
        keep_users = set(counts[counts >= (self.k + 1)].index.tolist())
        rx = rx.loc[rx["userId"].isin(keep_users)].reset_index(drop=True)
        if rx.empty:
            raise SystemExit(f"No users with ≥ {self.k + 1} distinct movies for"
                             f" last-{self.k} holdout.")

        train_idx, valid_idx = [], []
        for uid, g in rx.groupby("userId", sort=True):
            g = g.sort_values(["datetime", "movieId", "genre"], kind="mergesort")
            # Compute a per-movie "end time" using the most recent rating timestamp.
            # Sorting by this end time (ascending) to hold out the K most recent
            # distinct movies for each user, regardless of how many (genre) rows each movie has.
            movie_end = (g.groupby("movieId")["datetime"].max()
                         .sort_values(kind="mergesort")) # stable tie-breaking by movieId/genre via outer sort
            mid_ordered = movie_end.index.to_numpy()
            if mid_ordered.size <= self.k:
                continue
            valid_mids = set(mid_ordered[-self.k:])
            mask = g["movieId"].isin(valid_mids)
            vi = g.loc[mask].index.to_numpy()
            ti = g.loc[~mask].index.to_numpy()
            if ti.size == 0:
                continue
            train_idx.append(ti)
            valid_idx.append(vi)

        if not train_idx:
            raise SystemExit(f"No users leave any training data after last-{self.k} movies holdout.")

        train = rx.loc[np.concatenate(train_idx)].reset_index(drop=True)
        valid = rx.loc[np.concatenate(valid_idx)].reset_index(drop=True)
        print(f"[Split] Users: {train['userId'].nunique()}  Held-out per user: {self.k}  "
              f"Train rows: {len(train)}  Valid rows: {len(valid)}")
        return train, valid
## HMM Run
class HMMBackend:
    """Wrapper around hmmlearn with uniform API regardless of categorical/multinomial."""
    def __init__(self, cfg: HMMConfig, vocab: Vocabulary):
        self.cfg = cfg
        self.vocab = vocab
        self.model = None
        self.is_categorical = None  # bool
        self.M = None               # number of tokens

    def fit(self, X_tokens: np.ndarray, lengths: List[int]) -> None:
        """Fit the HMM using token sequences.
            Uses CategoricalHMM if available; otherwise falls back to
            MultinomialHMM with one-hot encoded observations.

            Args: X_tokens: Integer token array of shape (N, 1).
                lengths: Per-user sequence lengths for hmmlearn.
            Side Effects: Trains self.model and sets self.is_categorical, self.M.
            Prints: Convergence diagnostics from hmmlearn's monitor_ (if present)."""
        start = time.time()

        self.M = int(X_tokens.max()) + 1
        verbose = self.cfg.verbose_hmm

        self.model = CategoricalHMM(n_components=self.cfg.n_states, n_iter=200,
                                    tol=1e-3, random_state=self.cfg.random_state,
                                    verbose=verbose, n_features=self.M,
                                    init_params="ste",
                                    transmat_prior=np.ones((self.cfg.n_states,
                                                            self.cfg.n_states)), #mild Dirichlet priors
                                    startprob_prior=np.ones(self.cfg.n_states), #mild Dirichlet priors
                                    )

        print(f"[HMM] Fitting CategoricalHMM({self.cfg.n_states}) on "
              f"{len(lengths)} sequences, {len(X_tokens)} tokens...")
        self.model.fit(X_tokens, lengths)
        self.is_categorical = True

        print("[HMM] Training complete.")
        if hasattr(self.model, "monitor_"):
            print(f"Converged? {getattr(self.model.monitor_, 'converged', None)} "
                  f"iters={getattr(self.model.monitor_, 'iter', None)} "
                  f"last_ll={getattr(self.model.monitor_, 'history', [None])[-1]}")

        end = time.time()
        elapsed = end - start
        print(f"[HMM] Training time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")

    def score(self, X_tokens: np.ndarray, lengths: List[int]) -> float:
        """Compute total log-likelihood of sequences under the model.
        Args: X_tokens: Integer token array of shape (N, 1).
                lengths: Per-user sequence lengths.
        Returns: Total log-likelihood (float)."""
        if self.is_categorical:
            return self.model.score(X_tokens, lengths)
        X_oh = SequenceBuilder.tokens_to_onehot(X_tokens, self.M)
        return self.model.score(X_oh, lengths)

    def decode_states(self, user_tokens: np.ndarray) -> np.ndarray:
        """Viterbi decode most likely hidden state sequence for a user.
            Args: user_tokens: 1D array of integer tokens.
            Returns: 1D array of hidden state indices."""
        if self.is_categorical:
            _, states = self.model.decode(user_tokens.reshape(-1, 1), algorithm="viterbi")
        else:
            X_oh = SequenceBuilder.tokens_to_onehot(user_tokens, self.M)
            _, states = self.model.decode(X_oh, algorithm="viterbi")
        return states

    def predict_next_obs_distribution(self, user_tokens: np.ndarray, token_backoff: Optional[np.ndarray] = None,
                                      alpha: float = 0.0) -> np.ndarray:
        """
        Predict P(next token). Optionally blend with a global token prior:
          next = (1 - alpha) * (A[last] @ B) + alpha * token_backoff
        token_backoff should be length-M and sum to 1.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        states = self.decode_states(user_tokens.reshape(-1))
        last_state = int(states[-1])
        next_obs = self.model.transmat_[last_state] @ self.model.emissionprob_  # (M,)

        if token_backoff is not None and alpha > 0.0:
            # Safe normalize
            s = next_obs.sum()
            if s > 0:
                next_obs = next_obs / s
            next_obs = (1.0 - alpha) * next_obs + alpha * token_backoff

        # final normalization for safety
        s2 = next_obs.sum()
        if s2 > 0:
            next_obs = next_obs / s2
        return next_obs

    def save_npz(self, path: Path) -> None:
        """Persist fitted HMM parameters and vocabulary metadata.
            Args: path: Output NPZ path."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        np.savez_compressed(path,
                            transmat=self.model.transmat_,
                            emissionprob=self.model.emissionprob_,
                            startprob=self.model.startprob_,
                            id_to_genre=self.vocab.id_to_genre,
                            token_to_tuple=self.vocab.token_to_tuple,
                            M=self.M, is_categorical=bool(self.is_categorical),
                            n_states=self.cfg.n_states,
                            )

## Candidate scorer

class CandidateScorer:
    """Ranks movies by aggregating genre mass across multi-genre titles."""
    def __init__(self, vocab: Vocabulary, rx_all: pd.DataFrame):
        self.vocab = vocab
        self.movies_long = rx_all[["movieId", "clean_title", "year", "genre"]].drop_duplicates()

    def genre_mass_from_tokens(self, next_obs_dist: np.ndarray) -> np.ndarray:
        """Compress token distribution into a per-genre mass vector.
        Each token is (genre_id, rating_bin). Summing over rating bins yields
        a per-genre probability mass, normalized to 1.
        Args: next_obs_dist: Length-M token probability vector.
        Returns: Length-G genre mass vector."""
        if self.vocab.token_to_tuple is None or self.vocab.id_to_genre is None:
            raise RuntimeError("Vocabulary not initialized.")
        G = len(self.vocab.id_to_genre)
        genre_mass = np.zeros(G, dtype=np.float64)
        for k, (g_id, _rb) in enumerate(self.vocab.token_to_tuple):
            genre_mass[g_id] += next_obs_dist[k]
        s = genre_mass.sum()
        if s > 0:
            genre_mass /= s
        return genre_mass

    def score_candidates(self, genre_mass: np.ndarray, seen_movie_ids: set[int]) -> pd.DataFrame:
        """Compute movie-level scores by summing genre masses per movie.
        Args: genre_mass: Length-G genre mass vector.
                seen_movie_ids: Movies to exclude (already rated in reference set).
        Returns: DataFrame with columns [movieId, clean_title, year, score]."""
        candidates = (
            self.movies_long.loc[~self.movies_long["movieId"]
            .isin(seen_movie_ids), ["movieId", "clean_title", "year", "genre"]]
            .dropna(subset=["genre"])
            .copy())
        if candidates.empty:
            return pd.DataFrame(columns=["movieId", "clean_title", "year", "score"])
        candidates["genre_id"] = candidates["genre"].map(self.vocab.genre_to_id).astype("Int64")
        candidates = candidates.dropna(subset=["genre_id"]).astype({"genre_id": "int64"})

        scores = candidates.groupby(["movieId", "clean_title", "year"])["genre_id"].apply(
            lambda gids: float(np.sum(genre_mass[gids.to_numpy()]))
        ).reset_index(name="score")
        return scores

    # Each token corresponds to (genre_id, rating_bin).
    # To score a movie, we first compress token probabilities to genre mass by
    # summing over rating bins, then sum the masses of the movie's genres.
    # This linear aggregation keeps the ranking consistent with the model's
    # next-observation probabilities while respecting multi-genre titles.

## Recommender

class Recommender:
    """User-facing API for top-N and top-1 recommendations."""
    def __init__(self, backend: HMMBackend, vocab: Vocabulary,
                 rx_train: pd.DataFrame, rx_all: pd.DataFrame,
                 user_year_pref: Optional[pd.Series] = None,):
        self.backend = backend
        self.vocab = vocab
        self.rx_train = rx_train
        self.rx_all = rx_all
        self.scorer = CandidateScorer(vocab, rx_all)
        self.user_year_pref = user_year_pref  # Series: index=userId, value=preferred year

    def _user_obs(self, rx_ref: pd.DataFrame, user_id: int) -> np.ndarray:
        u = rx_ref.loc[rx_ref["userId"] == user_id]
        if u.empty or len(u) < 2:
            raise ValueError(f"User {user_id} has too few events.")
        return u["token"].to_numpy(dtype=np.int32)

    def _user_year_pref(self, user_id: int) -> Optional[float]:
        """Return the user's preferred release year, if available."""
        if self.user_year_pref is None:
            return None
        if user_id not in self.user_year_pref.index:
            return None
        return float(self.user_year_pref.loc[user_id])

    def _user_genre_prior(self, user_id: int, use_train_only: bool = True) -> np.ndarray:
        """
        Build a recency-weighted genre histogram for the user.

        More recent ratings get higher weight via exponential decay with
        a chosen half-life (in days).
        """
        rx_ref = self.rx_train if use_train_only else self.rx_all

        # Need both genre_id and datetime
        u = rx_ref.loc[rx_ref["userId"] == user_id, ["genre_id", "datetime"]]
        G = len(self.vocab.id_to_genre)
        if u.empty:
            # Fallback to uniform if we know nothing
            return np.full(G, 1.0 / G, dtype=float)

        # Ensure datetime is proper dtype
        u = u.copy()
        u["datetime"] = pd.to_datetime(u["datetime"], utc=True)

        # --- Recency weighting via exponential decay ---
        # Half-life: how many days until a rating's weight halves.
        half_life_days = 180.0  # tweak as you like or move to config
        half_life_seconds = half_life_days * 24 * 3600.0

        max_t = u["datetime"].max()
        age = (max_t - u["datetime"]).dt.total_seconds().to_numpy()

        # decay_rate * age = ln(2) * (age / half_life)
        decay_rate = np.log(2.0) / half_life_seconds
        weights = np.exp(-decay_rate * age)

        h = np.bincount(
            u["genre_id"].to_numpy(dtype=int),
            weights=weights,
            minlength=G,
        ).astype(float)

        if h.sum() > 0:
            h /= h.sum()
        else:
            h[:] = 1.0 / G
        return h

    def predict_genre_mass(self, user_id: int, use_train_only: bool = True,
                           token_backoff: Optional[np.ndarray] = None,
                           alpha_backoff: float = 0.0,
                           gamma_user: float = 0.0) -> np.ndarray:
        rx_ref = self.rx_train if use_train_only else self.rx_all
        u = rx_ref[rx_ref["userId"] == user_id]
        hist_len = len(u)

        tokens = self._user_obs(rx_ref, user_id)
        next_obs = self.backend.predict_next_obs_distribution(...)

        genre_mass = self.scorer.genre_mass_from_tokens(next_obs)

        if gamma_user > 0.0:
            user_prior = self._user_genre_prior(user_id, use_train_only=use_train_only)

            # NEW: scale gamma_user by history length (e.g., saturate at 500 rows)
            max_hist = 500.0
            scale = min(1.0, hist_len / max_hist)
            eff_gamma = gamma_user * scale

            genre_mass = (1.0 - eff_gamma) * genre_mass + eff_gamma * user_prior
            s = genre_mass.sum()
            if s > 0:
                genre_mass /= s
        return genre_mass

    def topN(self, user_id: int, N: int, use_train_only: bool = True,
             token_backoff: Optional[np.ndarray] = None,
             alpha_backoff: float = 0.0,
             gamma_user: float = 0.0,
             movie_pop: Optional[pd.Series] = None,
             beta_item_pop: float = 0.0,
             min_candidate_pop: int = 0,
             delta_year: float = 0.0) -> pd.DataFrame:
        """Return top-N recommended movies for a user.
                    Args: user_id: Target user id.
                        N: Number of movies to return.
                        use_train_only: Use only training data when True.
                    Returns: DataFrame w/ columns [movieId, clean_title, year, score], sorted desc by score.
                    Raises: ValueError: If the user has fewer than 2 events in the reference set.
                    """

        rx_ref = self.rx_train if use_train_only else self.rx_all
        u = rx_ref.loc[rx_ref["userId"] == user_id]
        tokens = self._user_obs(rx_ref, user_id)



        # blended genre mass
        genre_mass = self.predict_genre_mass(
            user_id,
            use_train_only=use_train_only,
            token_backoff=token_backoff,
            alpha_backoff=alpha_backoff,
            gamma_user=gamma_user,
        )

        seen = self._seen_cache.get(user_id, set(u["movieId"].unique()))
        scores = self.scorer.score_candidates(genre_mass, seen)

        # optional: filter ultra-rare --> (use counts)
        if min_candidate_pop > 0 and hasattr(self, "movie_pop_counts_default"):
            scores = scores.merge(self.movie_pop_counts_default.rename("pop_count"),
                                  left_on="movieId", right_index=True, how="left")
            scores["pop_count"] = scores["pop_count"].fillna(0).astype(int)
            scores = scores[scores["pop_count"] >= min_candidate_pop]

        # optional: blend item popularity
        # optional: blend item popularity
        if movie_pop is not None and beta_item_pop > 0.0:
            if "pop" not in scores.columns:
                scores = scores.merge(movie_pop.rename("pop"), left_on="movieId", right_index=True, how="left")
                scores["pop"] = scores["pop"].fillna(0.0)
            scores["score"] = (1.0 - beta_item_pop) * scores["score"] + beta_item_pop * scores["pop"]

        # optional: blend user release-year preference
        if delta_year > 0.0 and "year" in scores.columns:
            y_pref = self._user_year_pref(user_id)
            if y_pref is not None:
                # Convert to float, compute distance to preferred year
                scores["year"] = scores["year"].astype("float64")
                year_diff = (scores["year"] - y_pref).abs()
                # Convert year difference into a similarity in [0,1] (10-year scale, tweakable)
                year_sim = np.exp(-year_diff / 10.0)
                scores["year_align"] = year_sim.fillna(0.5)  # neutral if missing year
                scores["score"] = (
                        (1.0 - delta_year) * scores["score"]
                        + delta_year * scores["year_align"]
                )

        return (scores.sort_values(["score", "movieId"], ascending=[False, True])
                .head(N).reset_index(drop=True))

    def top1(self, user_id: int, use_train_only: bool = True) -> Optional[dict]:
        """Return a single best recommendation as a small dict.
        Keys: userId, movieId, clean_title, year, score. """
        df = self.topN(user_id, 1, use_train_only=use_train_only)
        if df.empty:
            return None
        r = df.iloc[0]
        return {
            "userId": int(user_id),
            "movieId": int(r["movieId"]),
            "clean_title": str(r["clean_title"]),
            "year": None if pd.isna(r["year"]) else int(r["year"]),
            "score": float(r["score"]),
        }
## Evaluation

class Evaluator:
    """Computes validation likelihood and recommendation metrics."""
    def __init__(self, backend: HMMBackend, seqs: SequenceBuilder, rx_train: pd.DataFrame, rx_valid: pd.DataFrame, rx_all: pd.DataFrame, recommender: Recommender):
        self.backend = backend
        self.seqs = seqs
        self.rx_train = rx_train
        self.rx_valid = rx_valid
        self.rx_all = rx_all
        self.recommender = recommender

    def validation_likelihood(self) -> Dict[str, float]:
        """Compute total and per-token validation log-likelihood + perplexity.
            Returns: Dict with keys {'loglike', 'avg_ll', 'perplexity'}. """
        X_valid, lengths_valid = self.seqs.to_X_lengths(self.rx_valid)
        ll = self.backend.score(X_valid, lengths_valid)
        avg_ll = ll / max(1, len(X_valid))
        perplexity = float(np.exp(-avg_ll))
        return {"loglike": float(ll), "avg_ll": float(avg_ll), "perplexity": perplexity}

    def rec_curves(self, Ks=(1, 5, 10, 20, 50)):
        recs = []
        for k in Ks:
            m = self.rec_metrics_at_k(k)
            recs.append((
                k,
                m[f"precision@{k}"],
                m[f"recall@{k}"],
            ))
        df = pd.DataFrame(recs, columns=["K", "precision", "recall"])
        return df

    def coverage_at_k(self, recommender, Ks=(1, 5, 10, 20, 50)):
        users = sorted(set(self.rx_train["userId"]) & set(self.rx_valid["userId"]))
        rec_items_allK = []
        for K in Ks:
            rec_items = set()
            for uid in users:
                try:
                    topk = recommender.topN(
                        uid, K, use_train_only=True,
                        token_backoff=getattr(recommender, "token_backoff_default", None),
                        alpha_backoff=getattr(recommender, "alpha_backoff_default", 0.0),
                        gamma_user=getattr(recommender, "gamma_user_default", 0.0),
                        movie_pop=getattr(recommender, "movie_pop_default", None),
                        beta_item_pop=getattr(recommender, "beta_item_pop_default", 0.0),
                        min_candidate_pop=getattr(recommender, "min_candidate_pop_default", 0),
                        delta_year=getattr(recommender, "delta_year_default", 0.0),
                    )

                    rec_items.update(topk["movieId"].tolist())
                except Exception:
                    pass
            rec_items_allK.append((K, len(rec_items)))
        return pd.DataFrame(rec_items_allK, columns=["K", "unique_recs"])

    def rec_metrics_at_k(self, k: int) -> Dict[str, float]:
        """Compute hit-rate@k, recall@k, and precision@k on the validation split.
        For each user:
          - Build top-k recommendations from the training history.
          - Compare against that user's held-out validation movies.
        Args: k: Cutoff for the recommendation list.

        Returns: Dict with keys:
              {'users', f'hit_rate@{k}', f'recall@{k}', f'precision@{k}'}.
        """
        users = sorted(set(self.rx_train["userId"]) & set(self.rx_valid["userId"]))

        hits, total = 0, 0
        sum_recall, sum_precision = 0.0, 0.0

        for uid in users:
            utrain = self.rx_train[self.rx_train["userId"] == uid]
            uvalid = self.rx_valid[self.rx_valid["userId"] == uid]
            if len(utrain) < 2 or uvalid.empty:
                continue

            try:
                topk = self.recommender.topN(
                    uid, k, use_train_only=True,
                    token_backoff=getattr(self.recommender, "token_backoff_default", None),
                    alpha_backoff=getattr(self.recommender, "alpha_backoff_default", 0.0),
                    gamma_user=getattr(self.recommender, "gamma_user_default", 0.0),
                    movie_pop=getattr(self.recommender, "movie_pop_default", None),
                    beta_item_pop=getattr(self.recommender, "beta_item_pop_default", 0.0),
                    min_candidate_pop=getattr(self.recommender, "min_candidate_pop_default", 0),
                    delta_year=getattr(self.recommender, "delta_year_default", 0.0),
                )

            except ValueError:
                continue

            heldout = set(uvalid["movieId"].unique())
            recs = set(topk["movieId"].tolist())
            inter = heldout & recs
            inter_size = len(inter)

            if not heldout:
                continue  # weird, but be safe

            total += 1
            # hit-rate: did we get at least one relevant item?
            hits += 1 if inter_size > 0 else 0

            # recall: fraction of held-out relevant items we recovered
            sum_recall += inter_size / max(1, len(heldout))

            # precision@k: fraction of recommended items that are relevant
            # If you want strict "per-user K", use max(1, k) here.
            denom = max(1, len(recs))  # handles cases where fewer than k are returned
            sum_precision += inter_size / denom

        return {
            "users": total,
            f"hit_rate@{k}": (hits / total) if total else 0.0,
            f"recall@{k}": (sum_recall / total) if total else 0.0,
            f"precision@{k}": (sum_precision / total) if total else 0.0,
        }
    def top1_last_movie_accuracy_table(self, user_filter: Optional[set[int]] = None) -> pd.DataFrame:
        """Evaluate whether the top-1 rec equals the last held-out movie per user.

                Args:
                    user_filter: If provided, restrict evaluation to these users.

                Returns:
                    Per-user DataFrame with columns:
                      ['userId', 'heldout_movieId', 'rec_movieId', 'rec_title',
                       'rec_score', 'hit', 'train_len'].
                """
        users = sorted(set(self.rx_train["userId"]) & set(self.rx_valid["userId"]))
        if user_filter is not None:
            users = [u for u in users if u in user_filter]
        rows = []
        for uid in users:
            uvalid = self.rx_valid[self.rx_valid["userId"] == uid]
            if uvalid.empty:
                continue
            gvalid = uvalid.sort_values(["datetime", "movieId", "genre"], kind="mergesort")
            heldout = int(gvalid.iloc[-1]["movieId"])
            try:
                rec = self.recommender.top1(uid, use_train_only=True)
            except ValueError:
                rows.append({"userId": uid, "heldout_movieId": heldout, "rec_movieId": None, "hit": False,
                             "train_len": int(self.rx_train[self.rx_train["userId"] == uid].shape[0])})
                continue
            if rec is None:
                rows.append({"userId": uid, "heldout_movieId": heldout, "rec_movieId": None, "hit": False,
                             "train_len": int(self.rx_train[self.rx_train["userId"] == uid].shape[0])})
                continue
            rows.append({
                "userId": uid,
                "heldout_movieId": heldout,
                "rec_movieId": rec["movieId"],
                "rec_title": rec["clean_title"],
                "rec_score": rec["score"],
                "hit": bool(rec["movieId"] == heldout),
                "train_len": int(self.rx_train[self.rx_train["userId"] == uid].shape[0]),
            })
        return pd.DataFrame(rows)

    def accuracy_by_history_threshold(self, thresholds: List[int], per_movie_history: bool = False) -> pd.DataFrame:
        """Aggregate top-1 accuracy for users with history >= each threshold.

                Args:
                    thresholds: Minimum history sizes to test.
                    per_movie_history: If True, history counts distinct movies (train only).
                        Otherwise counts exploded rows (train only).

                Returns:
                    DataFrame with columns ['threshold', 'users', 'hits', 'accuracy'].
                """
        hist = (self.rx_train.groupby("userId")["movieId"].nunique()
                if per_movie_history else self.rx_train.groupby("userId").size())
        results = []
        for t in sorted(set(thresholds)):
            elig = set(hist[hist >= t].index.tolist())
            table = self.top1_last_movie_accuracy_table(user_filter=elig)
            if table.empty:
                results.append({"threshold": t, "users": 0, "hits": 0, "accuracy": 0.0})
            else:
                users = table.shape[0]
                hits = int(table["hit"].sum())
                results.append({"threshold": t, "users": users, "hits": hits, "accuracy": float(hits / users)})
        return pd.DataFrame(results)

    def suggest_movie_thresholds(self, min_users: int = 50) -> List[int]:
        """Suggest thresholds based on quantiles of per-user train movie counts.

                Filters out thresholds that leave fewer than `min_users` eligible users.
                """
        hist = self.rx_train.groupby("userId")["movieId"].nunique()
        qs = [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95]
        raw = sorted({int(hist.quantile(q)) for q in qs})
        thresholds = [t for t in raw if t >= 5]
        keep = []
        for t in thresholds:
            users = int((hist >= t).sum())
            if users >= min_users:
                keep.append(t)
        return keep
## Orchestration/Run
class Pipeline:
    """Runs end-to-end training, evaluation, and artifact export."""

    def __init__(self, cfg: HMMConfig):
        self.cfg = cfg
        self.cfg.outdir.mkdir(exist_ok=True)
        self.vocab = Vocabulary(cfg.rating_bins)
        self.seq_builder = SequenceBuilder(self.vocab)
        self.backend = HMMBackend(cfg, self.vocab)

        self.movie_pop_norm: Optional[pd.Series] = None
        self.token_pop: Optional[np.ndarray] = None
        self.user_year_pref: Optional[pd.Series] = None


        # Dataframes
        self.rx_all: Optional[pd.DataFrame] = None
        self.rx_train: Optional[pd.DataFrame] = None
        self.rx_valid: Optional[pd.DataFrame] = None

    def _tune_n_states(self, candidates=(4, 6, 8, 10)) -> int:
        """Pick n_states by lowest validation perplexity over a small grid."""
        best = (None, float("inf"))
        X_train, lengths_train = self.seq_builder.to_X_lengths(self.rx_train)
        X_valid, lengths_valid = self.seq_builder.to_X_lengths(self.rx_valid)
        for s in candidates:
            tmp_cfg = dataclasses.replace(self.cfg, n_states=s)
            tmp_backend = HMMBackend(tmp_cfg, self.vocab)
            tmp_backend.fit(X_train, lengths_train)
            ll = tmp_backend.score(X_valid, lengths_valid)
            avg_ll = ll / max(1, len(X_valid))
            perp = float(np.exp(-avg_ll))
            if perp < best[1]:
                best = (s, perp)
            print(f"[tune] n_states={s}  perplexity={perp:.3f}")
        print(f"[tune] chosen n_states={best[0]}  perplexity={best[1]:.3f}")
        return best[0]

    def _tune_gamma_user(
            self,
            evaluator: "Evaluator",
            recommender: "Recommender",
            k: Optional[int] = None,
    ) -> None:
        """
        Tune gamma_user_genre (blend between HMM next-genre mass and
        recency-weighted user genre prior) using recall@k on validation.

        Updates:
          - self.cfg.gamma_user_genre
          - recommender.gamma_user_default
        """
        if k is None:
            k = self.cfg.top_n

        gammas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

        best_score = -1.0
        best_gamma = None

        for g in gammas:
            recommender.gamma_user_default = g

            metrics = evaluator.rec_metrics_at_k(k)
            score = metrics.get(f"recall@{k}", 0.0)

            print(f"[tune-gamma] gamma={g:.3f} -> recall@{k}={score:.4f}")

            if score > best_score:
                best_score = score
                best_gamma = g

        if best_gamma is not None:
            print(
                f"[tune-gamma] BEST gamma={best_gamma:.3f} "
                f"-> recall@{k}={best_score:.4f}"
            )
            self.cfg.gamma_user_genre = best_gamma
            recommender.gamma_user_default = best_gamma

    def _tune_recommender_hparams(
            self,
            evaluator: "Evaluator",
            recommender: "Recommender",
            k: Optional[int] = None,
    ) -> None:
        """
        Grid-search over recommender hyperparameters using recall@k
        on the validation split.
        Tunes:
          - alpha_pop_backoff  (token-level popularity backoff)
          - beta_item_pop      (item popularity blending)
          - min_candidate_pop  (min train count to be a candidate)
          - delta_year         (user release-year preference blending)

        gamma_user_genre is tuned separately in _tune_gamma_user.
        """
        if k is None:
            k = self.cfg.top_n

        # Search grids
        alphas = [0.0, 0.05, 0.10, 0.20]
        betas = [0.0, 0.10, 0.20, 0.40]
        mins = [0, 3, 5, 10]
        deltas = [0.0, 0.10, 0.20, 0.30]

        # Keep gamma fixed to current config
        fixed_gamma = self.cfg.gamma_user_genre

        best_score = -1.0
        best_cfg: Optional[Tuple[float, float, int, float]] = None

        for a in alphas:
            for b in betas:
                for m in mins:
                    for d in deltas:
                        # update defaults used by Evaluator.rec_metrics_at_k()
                        recommender.alpha_backoff_default = a
                        recommender.beta_item_pop_default = b
                        recommender.gamma_user_default = fixed_gamma
                        recommender.min_candidate_pop_default = m
                        recommender.delta_year_default = d

                        metrics = evaluator.rec_metrics_at_k(k)
                        score = metrics[f"recall@{k}"]

                        print(
                            f"[tune-rec] alpha={a:.3f} beta={b:.3f} "
                            f"gamma={fixed_gamma:.3f} min_pop={m} delta_year={d:.3f} "
                            f"-> recall@{k}={score:.4f}"
                        )

                        if score > best_score:
                            best_score = score
                            best_cfg = (a, b, m, d)

        if best_cfg is not None:
            a, b, m, d = best_cfg
            print(
                f"[tune-rec] BEST alpha={a:.3f} beta={b:.3f} "
                f"gamma={fixed_gamma:.3f} min_pop={m} delta_year={d:.3f} "
                f"-> recall@{k}={best_score:.4f}"
            )

            # Persist into config
            self.cfg.alpha_pop_backoff = a
            self.cfg.beta_item_pop = b
            self.cfg.min_candidate_pop = m
            self.cfg.delta_year = d

            # And into recommender defaults
            recommender.alpha_backoff_default = a
            recommender.beta_item_pop_default = b
            recommender.min_candidate_pop_default = m
            recommender.gamma_user_default = fixed_gamma
            recommender.delta_year_default = d

    def make_all_plots(self, evaluator: "Evaluator") -> None:
        """
        Save all figures/CSVs for the report in one place.
        Requires: self.backend fitted, evaluator.recommender set.
        """
        out = self.cfg.outdir
        out.mkdir(exist_ok=True)

        #  1) Training convergence (if available)
        hist = getattr(self.backend.model.monitor_, "history", None)
        if hist:
            plt.figure(figsize=(6.5, 3.6))
            plt.plot(hist)
            plt.xlabel("EM iteration");
            plt.ylabel("Total log-likelihood")
            plt.title("HMM training convergence")
            plt.tight_layout();
            plt.savefig(out / "plot_convergence.png", dpi=150);
            plt.close()

        #  2) Transition matrix heatmap
        A = self.backend.model.transmat_
        plt.figure(figsize=(5.6, 4.8))
        plt.imshow(A, aspect="auto")
        plt.colorbar(label="P(next state | current state)")
        plt.xlabel("next state");
        plt.ylabel("current state")
        plt.title("HMM transition matrix")
        plt.tight_layout();
        plt.savefig(out / "plot_transition_matrix.png", dpi=150);
        plt.close()

        # 3) State → genre emission mass
        B = self.backend.model.emissionprob_  # S x M
        tok = self.vocab.token_to_tuple  # (M, 2) -> (genre_id, rating_bin)
        G = len(self.vocab.id_to_genre)
        S, M = B.shape
        state_genre = np.zeros((S, G), dtype=float)
        for k, (g_id, rb) in enumerate(tok):
            state_genre[:, g_id] += B[:, k]
        # normalize rows
        row_sums = np.maximum(state_genre.sum(1, keepdims=True), 1e-12)
        state_genre = state_genre / row_sums

        plt.figure(figsize=(min(14, 0.22 * G + 4), 4.8))
        plt.imshow(state_genre, aspect="auto")
        plt.colorbar(label="P(genre | state)")
        plt.yticks(range(S), [f"s{i}" for i in range(S)])
        plt.title("State → genre emission mass")
        plt.tight_layout();
        plt.savefig(out / "plot_state_genre_emissions.png", dpi=150);
        plt.close()

        #  4) Coverage vs K
        cov_df = evaluator.coverage_at_k(evaluator.recommender, Ks=(1, 5, 10, 20, 50))
        cov_df.to_csv(out / "coverage_curve.csv", index=False)
        plt.figure(figsize=(6.5, 3.6))
        plt.plot(cov_df["K"], cov_df["unique_recs"], marker="o")
        plt.xlabel("K");
        plt.ylabel("# unique recommended items")
        plt.title("Catalog coverage vs K")
        plt.grid(True, alpha=0.3)
        plt.tight_layout();
        plt.savefig(out / "plot_coverage_curve.png", dpi=150);
        plt.close()

        # 5) Recall/Hit-rate curves
        curve_df = evaluator.rec_curves(Ks=(1, 5, 10, 20, 50, 100))
        curve_df.to_csv(out / "rec_curves.csv", index=False)
        plt.figure(figsize=(6.5, 3.6))
        plt.plot(curve_df["K"], curve_df["recall"], marker="o", label="Recall@K")
        plt.plot(curve_df["K"], curve_df["precision"], marker="s", label="Precision@K")
        plt.xlabel("K")
        plt.ylabel("score")
        plt.title("Recommendation curves")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "plot_rec_curves.png", dpi=150)
        plt.close()

        # 6) Accuracy vs history (movies / rows)
        thresholds = [500, 1000, 1500, 2000, 2500, 3000]
        acc_movies_df = evaluator.accuracy_by_history_threshold(thresholds, per_movie_history=True)
        acc_rows_df = evaluator.accuracy_by_history_threshold(thresholds, per_movie_history=False)
        acc_movies_df.to_csv(out / "accuracy_by_movies_1.csv", index=False)
        acc_rows_df.to_csv(out / "accuracy_by_rows_1.csv", index=False)

        # Bar plots
        if not acc_movies_df.empty:
            plt.figure(figsize=(6.8, 3.6))
            plt.plot(acc_movies_df["threshold"], acc_movies_df["accuracy"], marker="o")
            plt.xlabel("Min distinct train movies");
            plt.ylabel("Top-1 accuracy")
            plt.title("Accuracy vs user history (by movies)")
            plt.grid(True, alpha=0.3);
            plt.tight_layout()
            plt.savefig(out / "plot_accuracy_vs_history_movies.png", dpi=150);
            plt.close()

        if not acc_rows_df.empty:
            plt.figure(figsize=(6.8, 3.6))
            plt.plot(acc_rows_df["threshold"], acc_rows_df["accuracy"], marker="o")
            plt.xlabel("Min train rows");
            plt.ylabel("Top-1 accuracy")
            plt.title("Accuracy vs user history (by rows)")
            plt.grid(True, alpha=0.3);
            plt.tight_layout()
            plt.savefig(out / "plot_accuracy_vs_history_rows.png", dpi=150);
            plt.close()

    def run_all(self, target_user_id: Optional[int] = None) -> None:
        """Execute the full workflow and write outputs to disk.

                Steps:
                  1) Load CSVs.
                  2) Build sequences/tokens.
                  3) Split train/valid (last-K movies per user).
                  4) Fit the HMM.
                  5) Validation likelihood & recommendation metrics.
                  6) Accuracy-vs-history tables (movies/rows).
                  7) Top-1 last-movie accuracy table.
                  8) Save top-N recommendations for a demo user.
                  9) Persist observations and HMM parameters to NPZ.

                Args:
                    target_user_id: If None, chooses the user with the largest train
                        history for the demo top-N export.
                """


        # Load
        ratings, movies_long = DataLoader(self.cfg.ratings_path, self.cfg.movies_long_path).load()


        # Build sequences/tokens
        self.rx_all = self.seq_builder.build(ratings, movies_long)

        # Split
        self.rx_train, self.rx_valid = HoldoutSplitter(self.cfg.holdout_last_k_movies).split(self.rx_all)

        # --- User release-year preferences from TRAIN ---
        # Example: simple per-user mean release year (you can later make this rating- or recency-weighted)
        train_years = self.rx_train.dropna(subset=["year"]).copy()
        if not train_years.empty:
            train_years["year"] = train_years["year"].astype(float)
            self.user_year_pref = train_years.groupby("userId")["year"].mean()
        else:
            self.user_year_pref = None

        # Optional quick tune
        self.cfg.n_states = self._tune_n_states(candidates=(4, 6, 8, 10, 12, 16))
        print(f"[Tuning] Selected n_states={self.cfg.n_states}")

        # --- Popularity tables from TRAIN ---
        train_counts_movies = self.rx_train.groupby("movieId").size()
        self.movie_pop_counts = train_counts_movies  # needed by min_candidate_pop filter
        movie_pop_norm = (train_counts_movies / train_counts_movies.sum()).astype(float)

        # global token popularity (for backoff)
        token_counts = self.rx_train["token"].value_counts().sort_index()
        M = int(token_counts.index.max()) + 1
        token_pop = np.zeros(M, dtype=float)
        token_pop[token_counts.index.to_numpy()] = token_counts.to_numpy(dtype=float)
        token_pop = token_pop / max(1.0, token_pop.sum())

        self.movie_pop_norm = movie_pop_norm
        self.token_pop = token_pop

        # Train
        X_train, lengths_train = self.seq_builder.to_X_lengths(self.rx_train)
        self.backend.fit(X_train, lengths_train)

        # Eval: validation likelihood
        evaluator = Evaluator(self.backend, self.seq_builder, self.rx_train, self.rx_valid, self.rx_all,
                              recommender=None)  # placeholder; will set after recommender created

        val = evaluator.validation_likelihood()
        print(f"VALID: loglike={val['loglike']:.2f}  avg/token={val['avg_ll']:.4f}  perplexity={val['perplexity']:.3f}")

        # Recommender (needs trained backend + data)
        recommender = Recommender( self.backend, self.vocab, self.rx_train,
                                   self.rx_all, user_year_pref=self.user_year_pref,
                                   )

        # Static/population stuff (not tuned)
        recommender.token_backoff_default = self.token_pop
        recommender.movie_pop_default = self.movie_pop_norm
        recommender.movie_pop_counts_default = self.movie_pop_counts

        # Initial defaults from config (will be overwritten by tuning)
        recommender.alpha_backoff_default = self.cfg.alpha_pop_backoff
        recommender.beta_item_pop_default = self.cfg.beta_item_pop
        recommender.gamma_user_default = self.cfg.gamma_user_genre
        recommender.min_candidate_pop_default = self.cfg.min_candidate_pop
        recommender.delta_year_default = self.cfg.delta_year

        # Wire into evaluator so rec_metrics_at_k() can use it during tuning
        evaluator.recommender = recommender

        # --- tune recommender hyperparameters on validation set (alpha, beta, min_pop) ---
        self._tune_recommender_hparams(evaluator, recommender, k=self.cfg.top_n)

        # --- then tune gamma_user_genre using the recency-weighted prior ---
        self._tune_gamma_user(evaluator, recommender, k=self.cfg.top_n)

        # Run and save all plots/CSVs using tuned hyperparameters
        self.make_all_plots(evaluator)

        # Validation recommendation metrics with tuned config
        recm = evaluator.rec_metrics_at_k(self.cfg.top_n)
        print(
            f"VALID Rec Metrics (tuned gamma): users={recm['users']}  "
            f"precision@{self.cfg.top_n}={recm[f'precision@{self.cfg.top_n}']:.3f}  "
            f"recall@{self.cfg.top_n}={recm[f'recall@{self.cfg.top_n}']:.3f}"
        )

        # Per-user top1 last movie accuracy table
        top1_results = evaluator.top1_last_movie_accuracy_table()
        top1_results.to_csv(self.cfg.outdir / "eval_top1_last_movie_1.csv", index=False)
        print(top1_results[top1_results["hit"] == False].head(10))

        # Choose a user for demo recommendation (largest training history)
        if target_user_id is None:
            lens = self.rx_train.groupby("userId").size().sort_values(ascending=False)
            target_user_id = int(lens.index[0])

        # Use tuned hyperparameters from cfg
        # Use tuned hyperparameters from cfg
        alpha = self.cfg.alpha_pop_backoff
        beta = self.cfg.beta_item_pop
        gamma = self.cfg.gamma_user_genre
        min_pop = self.cfg.min_candidate_pop
        delta_year = self.cfg.delta_year

        recs = recommender.topN(target_user_id, self.cfg.top_n, use_train_only=True,
                                token_backoff=self.token_pop, alpha_backoff=alpha,
                                gamma_user=gamma, movie_pop=self.movie_pop_norm,
                                beta_item_pop=beta, min_candidate_pop=min_pop,
                                delta_year=delta_year,
                                )

        out_path = self.cfg.outdir / f"user_topN_{target_user_id}.csv"
        recs.to_csv(out_path, index=False)

        # Persist observations
        np.savez_compressed(
            self.cfg.outdir / "observations_1.npz",
            X=X_train,
            lengths=np.array(lengths_train, dtype=object),
            user_index=self.rx_train["userId"].drop_duplicates().to_numpy(),
            id_to_genre=self.vocab.id_to_genre,
        )

        # Save model
        self.backend.save_npz(self.cfg.outdir / "hmm_model.npz")

        print(f"Saved recommendations for user {target_user_id} → {out_path}")

# Main
if __name__ == "__main__":
    cfg = HMMConfig(
        n_states=6,
        rating_bins=(2.0, 3.0, 3.5, 4.0),    # 5 bins: 0: r <= 2.0 (strong dislike); 1: 2.0 < r <= 3.0(dislike / low) 2: 3.0 < r <= 3.5 (meh); 3: 3.5 < r <= 4.0 (like); 4: r > 4.0 (strong like)
        top_n=20,
        outdir=Path("out_hmm"),
        random_state=0,
        verbose_hmm=False,
        holdout_last_k_movies=100,
    )
    Pipeline(cfg).run_all(target_user_id=None)

