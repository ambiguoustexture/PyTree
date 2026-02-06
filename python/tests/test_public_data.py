from pathlib import Path

import pandas as pd
import pytest


ROOT = Path("python/tests/data/P-Tree-Public-Data-main")
FOLDERS = [
    "Train_1981_2020",
    "Train_1981_2000_Test_2001_2020",
    "Train_2001_2020_Test_1981_2000",
]


@pytest.fixture(params=FOLDERS)
def folder_path(request):
    path = ROOT / request.param
    if not path.exists():
        pytest.skip(f"Public data folder not found: {path}")
    return path


def test_public_data_not_empty(folder_path):
    csvs = sorted(folder_path.glob("*.csv"))
    assert csvs, f"No CSV files in {folder_path}"


def test_public_data_no_nan(folder_path):
    csvs = sorted(folder_path.glob("*.csv"))
    for csv in csvs:
        df = pd.read_csv(csv)
        drop_cols = [c for c in df.columns if c in ("Unnamed: 0", "")]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        assert not df.empty, f"Empty data in {csv}"
        numeric = df.select_dtypes(include="number")
        assert not numeric.isna().any().any(), f"NaN values in {csv}"


def main():
    if not ROOT.exists():
        raise FileNotFoundError(
            "Public data not found. Place the dataset at python/tests/data/P-Tree-Public-Data-main"
        )

    for folder in FOLDERS:
        folder_path = ROOT / folder
        if not folder_path.exists():
            raise FileNotFoundError(f"Missing folder: {folder_path}")
        csvs = sorted(folder_path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files in {folder_path}")
        for csv in csvs:
            df = pd.read_csv(csv)
            drop_cols = [c for c in df.columns if c in ("Unnamed: 0", "")]
            if drop_cols:
                df = df.drop(columns=drop_cols)
            if df.empty:
                raise ValueError(f"Empty data in {csv}")
            numeric = df.select_dtypes(include="number")
            if numeric.isna().any().any():
                raise ValueError(f"NaN values in {csv}")


if __name__ == "__main__":
    main()
