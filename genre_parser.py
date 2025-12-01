#!/usr/bin/env python3
"""
Parse hmm_training_movies.csv into long + wide genre formats and extract year.

Input:
  hmm_training_movies.csv  # columns: movieId,title,genres

Outputs (in ./out):
  movies_long.csv   # (movieId, clean_title, year, genre)
  movies_wide.csv   # (movieId, clean_title, year, <one column per genre>)
"""

import os
import re
from pathlib import Path
import pandas as pd

# Regex to extract the final "(YYYY)" from a title
YEAR_RX = re.compile(r"\((\d{4})\)\s*$")

def extract_year(title: str):
    """Return (clean_title, year) for a movie title."""
    if not isinstance(title, str):
        return title, None
    m = YEAR_RX.search(title)
    year = int(m.group(1)) if m else None
    clean = YEAR_RX.sub("", title).strip() if m else title.strip()
    return clean, year

def main():
    # Define input and output
    input_csv = "hmm_training_movies.csv"
    outdir = Path("../out")
    outdir.mkdir(exist_ok=True)

    if not Path(input_csv).exists():
        raise FileNotFoundError(f"Could not find {input_csv} in the current directory.")

    print(f"📂 Reading {input_csv} ...")
    df = pd.read_csv(
        input_csv,
        dtype={"movieId": "int64", "title": "string", "genres": "string"}
    )

    # Normalize and extract year
    df["genres"] = df["genres"].fillna("(no genres listed)").astype(str)
    clean_titles, years = zip(*df["title"].map(extract_year))
    df["clean_title"] = pd.Series(clean_titles, dtype="string")
    df["year"] = pd.Series(years, dtype="Int64")

    # Long format (one row per movie/genre)
    long_df = (
        df.assign(genre=df["genres"].str.split("|"))
          .explode("genre", ignore_index=True)
          .assign(genre=lambda d: d["genre"].fillna("(no genres listed)"))
          [["movieId", "clean_title", "year", "genre"]]
          .sort_values(["movieId", "genre"], kind="mergesort")
          .reset_index(drop=True)
    )
    long_path = outdir / "movies_long.csv"
    long_df.to_csv(long_path, index=False)

    # Wide format (multi-hot genre columns)
    genres = sorted(long_df["genre"].dropna().unique().tolist())
    wide_df = (
        long_df.assign(value=1)
               .pivot_table(index=["movieId", "clean_title", "year"],
                            columns="genre", values="value", fill_value=0)
               .reset_index()
               .rename_axis(None, axis=1)
    )
    for g in genres:
        if g not in wide_df.columns:
            wide_df[g] = 0
    wide_df = wide_df[["movieId", "clean_title", "year"] + genres]
    wide_path = outdir / "movies_wide.csv"
    wide_df.to_csv(wide_path, index=False)

    print(f"✅ Parsing complete!")
    print(f"  - Long format saved to: {long_path}")
    print(f"  - Wide format saved to: {wide_path}")
    print(f"  - Found {len(genres)} genres: {', '.join(genres)}")

if __name__ == "__main__":
    main()
