from pathlib import Path

import numpy as np
import pandas as pd

root = Path("python/tests/data/P-Tree-Public-Data-main")
out_file = Path("python/tests/data/public_data_stats_py.csv")

folders = [
    "Train_1981_2020",
    "Train_1981_2000_Test_2001_2020",
    "Train_2001_2020_Test_1981_2000",
]


def read_stats(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    drop_cols = [c for c in df.columns if c in ("Unnamed: 0", "")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = df.select_dtypes(include="number")
    stats = []
    for col in df.columns:
        x = df[col].to_numpy()
        mean = float(np.mean(x))
        sd = float(np.std(x, ddof=1))
        sharpe = 0.0 if sd == 0 else mean / sd
        stats.append(
            dict(
                file=path.name,
                column=col,
                mean=mean,
                sd=sd,
                sharpe=sharpe,
                n=len(df),
            )
        )
    return pd.DataFrame(stats)


def main() -> None:
    if not root.exists():
        raise FileNotFoundError(
            "Public data not found. Place the dataset at python/tests/data/P-Tree-Public-Data-main"
        )

    all_stats = []
    for folder in folders:
        folder_path = root / folder
        csvs = sorted(folder_path.glob("*.csv"))
        for csv in csvs:
            stats = read_stats(csv)
            stats["folder"] = folder
            all_stats.append(stats)

    if not all_stats:
        raise RuntimeError("No CSV files found.")

    all_stats_df = pd.concat(all_stats, ignore_index=True)
    all_stats_df.to_csv(out_file, index=False)
    print(f"Wrote {out_file}")

    r_file = Path("python/tests/data/public_data_stats_r.csv")
    if not r_file.exists():
        print("R stats not found; skipping comparison")
        return

    r_stats = pd.read_csv(r_file)
    merged = all_stats_df.merge(
        r_stats, on=["folder", "file", "column"], suffixes=("_py", "_r"), how="inner"
    )
    if merged.empty:
        raise RuntimeError("No overlapping rows between R and Python stats")

    for col in ["mean", "sd", "sharpe"]:
        diff = (merged[f"{col}_py"] - merged[f"{col}_r"]).abs().max()
        if diff > 1e-6:
            raise AssertionError(f"{col} mismatch; max diff {diff}")

    print("R vs Python stats: OK")


if __name__ == "__main__":
    main()
