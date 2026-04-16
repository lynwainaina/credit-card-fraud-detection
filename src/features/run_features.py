"""
Feature engineering pipeline runner.

Loads data/cleaned.csv, engineers new features, selects the best subset,
saves the result to data/features.csv, and prints a summary report.
"""

import time
import os
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from src.data.loader import load_csv
from src.features.engineering import create_features, select_features


DATA_DIR = Path(__file__).parents[2] / "data"
load_dotenv(Path(__file__).parents[2] / ".env")

features_file = os.getenv("FEATURES_FILE")  
TARGET_COL = os.getenv("TARGET_COL")

def main(file_name) -> None:
    t0 = time.time()
    # laod the cleaned data and return tprint the shape
    print(f"Loading file {file_name} ...")
    df = load_csv(file_name)
    print(f"  Input shape:  {df.shape[0]:,} rows x {df.shape[1]} columns")
    # Engineer features
    print("\nRunning create_features for feature engineering...")
    engineered = create_features(df)
    new_cols = [c for c in engineered.columns if c not in df.columns]
    print(f"  After engineering: {engineered.shape[0]:,} rows x {engineered.shape[1]} columns "
          f"(+{len(new_cols)} new features)")
    # Select relevant features — exclude Class (target) so it is never filtered out
    print("\nRunning select_features ...")
    feature_df = engineered.drop(columns=[TARGET_COL])
    selected, reduced = select_features(feature_df)
    if TARGET_COL in engineered.columns:
        reduced = pd.concat([reduced, engineered[[TARGET_COL]]], axis=1)
        selected = selected + [TARGET_COL]
    dropped = [c for c in engineered.columns if c not in selected]
    print(f"  After selection:   {reduced.shape[0]:,} rows x {reduced.shape[1]} columns "
          f"({len(dropped)} dropped)")
    print(f"\nKept features ({len(selected)}):")
    for col in selected:
        print(f"  {col}")

    if dropped:
        print(f"\nDropped features ({len(dropped)}):")
        for col in dropped:
            print(f"  {col}")

    # Save the engineered features to a csv file
    out_path = DATA_DIR / features_file
    reduced.to_csv(out_path, index=False)
    print(f"\nSaved {reduced.shape[0]:,} rows x {reduced.shape[1]} columns -> {out_path}")
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run feature engineering on a cleaned CSV dataset from data/")                                 
    parser.add_argument("filename", nargs="?", default="cleaned.csv",
                        help="CSV filename inside data. Defaults to cleaned.csv.")                  
    args = parser.parse_args()
    main(args.filename)
