import pandas as pd
from pathlib import Path
from typing import List, Set, Dict


class DatasetSubsetter:
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize the DatasetSubsetter with input and output directories.

        Args:
            data_dir (str): Directory containing the original CSV files
            output_dir (str): Directory where subset files will be saved
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.links_file = self.data_dir / "links.csv"
        self.movies_file = self.data_dir / "movies.csv"
        self.ratings_file = self.data_dir / "ratings.csv"
        self.tags_file = self.data_dir / "tags.csv"
        self.genome_scores_file = self.data_dir / "genome-scores.csv"
        self.genome_tags_file = self.data_dir / "genome-tags.csv"
        self.users_selection_file = self.data_dir / "users_selection.csv"
        self.movies_selection_file = self.data_dir / "movies_selection.csv"

        # Data storage
        self.selected_user_ids: Set[int] = set()
        self.selected_movie_ids: Set[int] = set()
        self.subset_data: Dict[str, pd.DataFrame] = {}

    # -----------------------
    # Utility / inspection
    # -----------------------
    def inspect_file_columns(self, file_path: Path) -> List[str]:
        """Inspect and return column names of a CSV file."""
        try:
            df_sample = pd.read_csv(file_path, nrows=5)
            print(f"Columns in {file_path.name}: {list(df_sample.columns)}")
            return list(df_sample.columns)
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            return []

    @staticmethod
    def _coerce_int_series(s: pd.Series) -> pd.Series:
        """Coerce a series to int64 safely."""
        return pd.to_numeric(s, errors="coerce").dropna().astype("int64")

    # -----------------------
    # Load selections
    # -----------------------
    def load_selections(self) -> None:
        """Load the user and movie selections from CSV files with flexible column names."""
        print("Loading user and movie selections...")

        # Inspect structures (helpful debug)
        print("\nInspecting file structures:")
        self.inspect_file_columns(self.users_selection_file)
        self.inspect_file_columns(self.movies_selection_file)

        # Load user selections with flexible column naming
        if self.users_selection_file.exists():
            users_df = pd.read_csv(self.users_selection_file)
            # Try different possible column names for user ID
            user_id_columns = ['userId', 'user_id', 'userID', 'user', 'UserID', 'User']
            user_id_col = next((c for c in user_id_columns if c in users_df.columns), None)

            if user_id_col is None:
                # Fall back to first column
                user_id_col = users_df.columns[0]
                print(f"⚠️ No standard user ID column found; using first column '{user_id_col}'")

            self.selected_user_ids = set(self._coerce_int_series(users_df[user_id_col]).tolist())
            print(f"Loaded {len(self.selected_user_ids)} selected users from column '{user_id_col}'")
        else:
            raise FileNotFoundError(f"Users selection file not found: {self.users_selection_file}")

        # Load movie selections with flexible column naming
        if self.movies_selection_file.exists():
            movies_df = pd.read_csv(self.movies_selection_file)
            # Try different possible column names for movie ID
            movie_id_columns = ['movieId', 'movie_id', 'movieID', 'movie', 'MovieID', 'Movie']
            movie_id_col = next((c for c in movie_id_columns if c in movies_df.columns), None)

            if movie_id_col is None:
                # Fall back to first column
                movie_id_col = movies_df.columns[0]
                print(f"⚠️ No standard movie ID column found; using first column '{movie_id_col}'")

            self.selected_movie_ids = set(self._coerce_int_series(movies_df[movie_id_col]).tolist())
            print(f"Loaded {len(self.selected_movie_ids)} selected movies from column '{movie_id_col}'")
        else:
            raise FileNotFoundError(f"Movies selection file not found: {self.movies_selection_file}")

        print(f"\nFinal selection counts → Users: {len(self.selected_user_ids)} | Movies: {len(self.selected_movie_ids)}")

    # -----------------------
    # Subset creators
    # -----------------------
    def create_links_subset(self) -> pd.DataFrame:
        """Create subset of links data based on selected movies."""
        print("Creating links subset...")
        links_df = pd.read_csv(self.links_file)
        links_df['movieId'] = self._coerce_int_series(links_df['movieId'])
        subset = links_df[links_df['movieId'].isin(self.selected_movie_ids)]
        print(f"Links subset: {len(subset)} rows (from {len(links_df)} total)")
        return subset

    def create_movies_subset(self) -> pd.DataFrame:
        """Create subset of movies data based on selected movies."""
        print("Creating movies subset...")
        movies_df = pd.read_csv(self.movies_file)
        movies_df['movieId'] = self._coerce_int_series(movies_df['movieId'])
        subset = movies_df[movies_df['movieId'].isin(self.selected_movie_ids)]
        print(f"Movies subset: {len(subset)} rows (from {len(movies_df)} total)")
        return subset

    def _ratings_reader(self, chunksize: int = 100_000):
        """Yield ratings chunks with safe dtypes."""
        usecols = ["userId", "movieId", "rating", "timestamp"]
        dtype = {"userId": "Int64", "movieId": "Int64", "rating": "float32", "timestamp": "Int64"}
        return pd.read_csv(self.ratings_file, usecols=usecols, dtype=dtype, chunksize=chunksize)

    def create_ratings_subset(self) -> pd.DataFrame:
        """
        Create subset of ratings based on selected users AND movies, with diagnostics.
        Returns empty DataFrame with proper columns if nothing matches.
        """
        print("Creating ratings subset (users ∩ movies)...")

        if not self.selected_user_ids or not self.selected_movie_ids:
            print("⚠️ Missing selections; run load_selections() first.")
            return pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])

        user_ids = set(int(x) for x in self.selected_user_ids)
        movie_ids = set(int(x) for x in self.selected_movie_ids)

        kept_chunks = []
        kept_rows = 0
        total_rows = 0
        present_selected_users = set()
        present_selected_movies = set()

        try:
            reader = self._ratings_reader()
        except FileNotFoundError:
            print(f"❌ ratings file not found: {self.ratings_file}")
            return pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])

        for i, chunk in enumerate(reader, start=1):
            chunk = chunk.dropna(subset=["userId", "movieId"]).copy()
            if chunk.empty:
                continue

            total_rows += len(chunk)
            chunk["userId"] = chunk["userId"].astype("int64")
            chunk["movieId"] = chunk["movieId"].astype("int64")

            present_selected_users.update(set(chunk["userId"].unique()) & user_ids)
            present_selected_movies.update(set(chunk["movieId"].unique()) & movie_ids)

            mask = chunk["userId"].isin(user_ids) & chunk["movieId"].isin(movie_ids)
            filtered = chunk.loc[mask]
            kept_rows += len(filtered)

            if i % 10 == 0 or len(filtered) > 0:
                print(f"Chunk {i}: kept {len(filtered)} / {len(chunk)} (cumulative kept: {kept_rows})")

            if not filtered.empty:
                kept_chunks.append(filtered)

        if kept_chunks:
            subset = pd.concat(kept_chunks, ignore_index=True)
            subset = subset.sort_values(["userId", "timestamp"], kind="mergesort").reset_index(drop=True)
        else:
            subset = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])

        print(f"Ratings scanned: {total_rows} | kept: {kept_rows}")
        if kept_rows == 0:
            if not present_selected_users and not present_selected_movies:
                print("🔎 None of the selected users or movies appear in ratings.csv.")
            elif not present_selected_users:
                print("🔎 Selected users do not appear in ratings.csv.")
            elif not present_selected_movies:
                print("🔎 Selected movies do not appear in ratings.csv.")
            else:
                print("🔎 Users and movies both appear, but no overlapping user–movie pairs.")

        return subset

    def create_ratings_user_subset(self) -> pd.DataFrame:
        """Ratings filtered by users only (for debugging overlap)."""
        print("Creating ratings_user_subset (users only)...")
        if not self.selected_user_ids:
            print("⚠️ No selected users.")
            return pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])

        user_ids = set(int(x) for x in self.selected_user_ids)
        kept_chunks = []
        kept_rows = 0
        total_rows = 0

        try:
            reader = self._ratings_reader()
        except FileNotFoundError:
            print(f"❌ ratings file not found: {self.ratings_file}")
            return pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])

        for i, chunk in enumerate(reader, start=1):
            chunk = chunk.dropna(subset=["userId"]).copy()
            if chunk.empty:
                continue
            total_rows += len(chunk)
            chunk["userId"] = chunk["userId"].astype("int64")
            filtered = chunk.loc[chunk["userId"].isin(user_ids)]
            kept_rows += len(filtered)
            if i % 10 == 0 or len(filtered) > 0:
                print(f"[users] Chunk {i}: kept {len(filtered)} / {len(chunk)} (cum: {kept_rows})")
            if not filtered.empty:
                kept_chunks.append(filtered)

        subset = (pd.concat(kept_chunks, ignore_index=True)
                  .sort_values(["userId", "timestamp"], kind="mergesort")
                  .reset_index(drop=True)) if kept_chunks else pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
        print(f"[users] Ratings scanned: {total_rows} | kept: {kept_rows}")
        return subset

    def create_ratings_movie_subset(self) -> pd.DataFrame:
        """Ratings filtered by movies only (for debugging overlap)."""
        print("Creating ratings_movie_subset (movies only)...")
        if not self.selected_movie_ids:
            print("⚠️ No selected movies.")
            return pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])

        movie_ids = set(int(x) for x in self.selected_movie_ids)
        kept_chunks = []
        kept_rows = 0
        total_rows = 0

        try:
            reader = self._ratings_reader()
        except FileNotFoundError:
            print(f"❌ ratings file not found: {self.ratings_file}")
            return pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])

        for i, chunk in enumerate(reader, start=1):
            chunk = chunk.dropna(subset=["movieId"]).copy()
            if chunk.empty:
                continue
            total_rows += len(chunk)
            chunk["movieId"] = chunk["movieId"].astype("int64")
            filtered = chunk.loc[chunk["movieId"].isin(movie_ids)]
            kept_rows += len(filtered)
            if i % 10 == 0 or len(filtered) > 0:
                print(f"[movies] Chunk {i}: kept {len(filtered)} / {len(chunk)} (cum: {kept_rows})")
            if not filtered.empty:
                kept_chunks.append(filtered)

        subset = (pd.concat(kept_chunks, ignore_index=True)
                  .sort_values(["movieId", "timestamp"], kind="mergesort")
                  .reset_index(drop=True)) if kept_chunks else pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
        print(f"[movies] Ratings scanned: {total_rows} | kept: {kept_rows}")
        return subset

    def create_tags_subset(self) -> pd.DataFrame:
        """Create subset of tags data based on selected users and movies."""
        print("Creating tags subset...")
        if not self.tags_file.exists():
            print("Tags file not found, skipping...")
            return pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])

        tags_df = pd.read_csv(self.tags_file)
        # Coerce types
        if "userId" in tags_df.columns:
            tags_df["userId"] = self._coerce_int_series(tags_df["userId"])
        if "movieId" in tags_df.columns:
            tags_df["movieId"] = self._coerce_int_series(tags_df["movieId"])

        subset = tags_df[
            tags_df['userId'].isin(self.selected_user_ids) &
            tags_df['movieId'].isin(self.selected_movie_ids)
        ]
        print(f"Tags subset: {len(subset)} rows (from {len(tags_df)} total)")
        return subset

    def create_genome_scores_subset(self) -> pd.DataFrame:
        """Create subset of genome scores based on selected movies (chunked)."""
        print("Creating genome scores subset...")
        if not self.genome_scores_file.exists():
            print("Genome scores file not found, skipping...")
            return pd.DataFrame(columns=["movieId", "tagId", "relevance"])

        kept_chunks = []
        kept_rows = 0
        total_rows = 0
        try:
            for i, chunk in enumerate(pd.read_csv(self.genome_scores_file, chunksize=100_000), start=1):
                if "movieId" not in chunk.columns:
                    print("⚠️ genome-scores.csv missing 'movieId' column.")
                    return pd.DataFrame(columns=["movieId", "tagId", "relevance"])
                chunk["movieId"] = self._coerce_int_series(chunk["movieId"])
                total_rows += len(chunk)
                filtered = chunk[chunk['movieId'].isin(self.selected_movie_ids)]
                kept_rows += len(filtered)
                if i % 10 == 0 or len(filtered) > 0:
                    print(f"[genome-scores] Chunk {i}: kept {len(filtered)} / {len(chunk)} (cum: {kept_rows})")
                if not filtered.empty:
                    kept_chunks.append(filtered)
        except FileNotFoundError:
            print(f"❌ genome-scores file not found: {self.genome_scores_file}")
            return pd.DataFrame(columns=["movieId", "tagId", "relevance"])

        subset = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else pd.DataFrame(columns=["movieId", "tagId", "relevance"])
        print(f"[genome-scores] scanned: {total_rows} | kept: {kept_rows}")
        return subset

    def create_genome_tags_subset(self) -> pd.DataFrame:
        """Create subset of genome tags based on tagIds present in genome-scores subset."""
        print("Creating genome tags subset...")
        if not self.genome_tags_file.exists():
            print("Genome tags file not found, skipping...")
            return pd.DataFrame(columns=["tagId", "tag"])

        genome_tags_df = pd.read_csv(self.genome_tags_file)
        if 'genome-scores' in self.subset_data and not self.subset_data['genome-scores'].empty:
            used_tag_ids = set(pd.to_numeric(self.subset_data['genome-scores']['tagId'], errors="coerce").dropna().astype("int64").unique())
            subset = genome_tags_df[genome_tags_df['tagId'].isin(used_tag_ids)]
            print(f"Genome tags subset: {len(subset)} rows (from {len(genome_tags_df)} total)")
        else:
            subset = genome_tags_df
            print("No genome scores to filter by, returning all genome tags")
        return subset

    # -----------------------
    # Orchestration
    # -----------------------
    def create_all_subsets(self) -> None:
        """Create all data subsets and store them in memory."""
        print("Starting to create all data subsets...")

        # Load selections first
        self.load_selections()

        # Warn if empty
        if not self.selected_user_ids and not self.selected_movie_ids:
            print("⚠️ No users or movies selected!")
            self.subset_data = {}
            return

        # Build subsets
        self.subset_data = {
            'links': self.create_links_subset(),
            'movies': self.create_movies_subset(),
            # ratings flavors:
            'ratings': self.create_ratings_subset(),               # users ∩ movies
            'ratings_user': self.create_ratings_user_subset(),     # users only
            'ratings_movie': self.create_ratings_movie_subset(),   # movies only
            'tags': self.create_tags_subset(),
            'genome-scores': self.create_genome_scores_subset(),
            'genome-tags': None,  # filled after genome-scores (needs tagIds)
        }
        # genome-tags depends on genome-scores
        self.subset_data['genome-tags'] = self.create_genome_tags_subset()

        print("All subsets created.")

    def save_subsets(self, prefix: str = "subset") -> None:
        """
        Save all subset data to CSV files.

        Args:
            prefix (str): Prefix for output filenames
        """
        print(f"Saving subsets with prefix '{prefix}'...")

        for name, df in self.subset_data.items():
            if df is None:
                continue
            if not isinstance(df, pd.DataFrame):
                continue
            output_file = self.output_dir / f"{prefix}_{name}.csv"
            if not df.empty:
                df.to_csv(output_file, index=False)
                print(f"Saved {output_file} with {len(df)} rows")
            else:
                # Still write header so downstream code has schema
                df.to_csv(output_file, index=False)
                print(f"Saved {output_file} (header only; 0 rows)")

    def get_subset_info(self) -> Dict[str, Dict]:
        """
        Get information about the created subsets.

        Returns:
            Dict containing information about each subset
        """
        info = {}
        for name, df in self.subset_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                info[name] = {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 ** 2
                }
            else:
                info[name] = {'rows': 0, 'columns': [], 'memory_usage_mb': 0.0}
        return info


def main():
    """Main function to run from PyCharm (no command line needed)."""
    # EDIT THESE PATHS FOR SPECIFIC MACHINE
    DATA_DIR = "/project_cs271"
    OUTPUT_DIR = "/project_cs271"

    print("MovieLens Dataset Subset Creator")
    print("=" * 60)
    print(f"Data dir   : {DATA_DIR}")
    print(f"Output dir : {OUTPUT_DIR}")

    subsetter = DatasetSubsetter(DATA_DIR, OUTPUT_DIR)

    try:
        subsetter.create_all_subsets()
        subsetter.save_subsets(prefix="hmm_training")

        print("\n" + "=" * 60)
        print("SUBSET CREATION SUMMARY")
        print("=" * 60)
        info = subsetter.get_subset_info()
        total_rows = 0
        for name, data in info.items():
            print(f"{name:15}: {data['rows']:7} rows, {len(data['columns']):2} cols, "
                  f"{data['memory_usage_mb']:7.2f} MB")
            total_rows += data['rows']

        print(f"\nTotal selected users : {len(subsetter.selected_user_ids)}")
        print(f"Total selected movies: {len(subsetter.selected_movie_ids)}")
        print(f"Total rows (all dfs) : {total_rows}")
        print("\nOutput files include:")
        print("  - hmm_training_ratings_user.csv (users only)")
        print("  - hmm_training_ratings_movie.csv (movies only)")
        print("  - hmm_training_ratings.csv (users ∩ movies)")

    except Exception as e:
        print(f"Error creating subsets: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
