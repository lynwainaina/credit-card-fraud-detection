import argparse
from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).parents[2] / "data"


def load_csv(filename: str) -> pd.DataFrame:
    """
    Load a CSV file from the data/ folder.
    
    Args:
        filename: CSV filename inside data/ (e.g. creditcard.csv).
    Returns:
        df: pandas DataFrame containing the data.
    """
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    return df


def print_shape(df: pd.DataFrame) -> None:
    """Print the shape of the DataFrame.
    
    Args:
        df: pandas DataFrame containing the data.
    Returns:
        None
    """
    rows, cols = df.shape
    print(f"Shape: {rows:,} rows x {cols} columns")


def print_dtypes(df: pd.DataFrame) -> None:
    """Print the column names and data types of the DataFrame.
    
    Args:
        df: pandas DataFrame containing the data.
    Returns:
        None
    """
    print("\nColumn names and data types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")


def print_summary_stats(df: pd.DataFrame) -> None:
    """Print the summary statistics of the DataFrame.
    
    Args:
        df: pandas DataFrame containing the data.
    Returns:
        None
    """
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        print("\nNo numeric columns found.")
        return
    stats = numeric.agg(["mean", "std", "min", "max"])
    print("\nSummary statistics (numeric columns):")
    print(stats.to_string())


def print_missing(df: pd.DataFrame) -> None:
    """Print the missing values of the DataFrame.
    
    Args:
        df: pandas DataFrame containing the data.
    Returns:
        None
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\nMissing values:")
    if missing.empty:
        print("  None")
        return
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"count": missing, "pct": pct})
    for col, row in report.iterrows():
        print(f"  {col}: {int(row['count']):,} ({row['pct']}%)")


def profile(filename: str) -> pd.DataFrame:
    """
    Profile a CSV dataset from the data/ folder.
    Args:
        filename: CSV filename inside data/ (e.g. creditcard.csv). Defaults to creditcard.csv.
    Returns:
        df: pandas DataFrame containing the data.
    Prints:
        Shape: 284,807 rows x 31 columns
        Column names and data types
        Summary statistics (numeric columns)
        Missing values: None
    """
    df = load_csv(filename)
    print_shape(df)
    print_dtypes(df)
    print_summary_stats(df)
    print_missing(df)
    return df


if __name__ == "__main__":
    # Create the argument parser. The description is what shows up when you run --help.
    parser = argparse.ArgumentParser(description="Profile a CSV dataset from the data/ folder")
    # nargs="?" makes filename optional. default="creditcard.csv" is used when nothing is passed.
    parser.add_argument("filename", nargs="?", default="creditcard.csv",
                        help="CSV filename inside data/ (e.g. creditcard.csv). Defaults to creditcard.csv.")
    # Read the terminal input and populates args. After this line, args.filename is either the value you typed or the default.
    args = parser.parse_args()
    filename = args.filename
    profile(filename)
