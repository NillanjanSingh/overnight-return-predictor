import numpy as np
import pandas as pd
from pathlib import Path


def get_daily_df(path: str) -> pd.DataFrame:
    daily_data_path = Path(path)
    d_dfs = []
    for file_path in daily_data_path.iterdir():
        if file_path.is_file() and file_path.suffix == ".parquet":
            temp_df = pd.read_parquet(file_path)
            temp_df["symbol"] = file_path.stem
            d_dfs.append(temp_df)
    return pd.concat(d_dfs, ignore_index=True)


def get_minute_df(path: str) -> pd.DataFrame:
    minute_data_path = Path(path)
    m_dfs = []
    for file_path in minute_data_path.iterdir():
        if file_path.is_file() and file_path.suffix == ".parquet":
            temp_df = pd.read_parquet(file_path)
            temp_df["symbol"] = file_path.stem
            m_dfs.append(temp_df)
    return pd.concat(m_dfs, ignore_index=True)


def build_actuals(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    df["target_date"] = g["date"].shift(-1)
    df["next_open"] = g["open"].shift(-1)
    df = df.dropna(subset=["target_date", "next_open"])
    df = df.rename(columns={"date": "pred_date"})
    df["actual_return_pct"] = (df["next_open"] / df["close"] - 1.0) * 100.0
    df["actual_direction"] = np.where(df["actual_return_pct"] < 0, -1, 1)
    df["actual_magnitude_pct"] = df["actual_return_pct"].abs()
    out = df[
        [
            "pred_date",
            "target_date",
            "symbol",
            "actual_return_pct",
            "actual_direction",
            "actual_magnitude_pct",
        ]
    ].copy()
    return out


def actuals_for_predictions(
    actuals_df: pd.DataFrame, predictions_df: pd.DataFrame
) -> pd.DataFrame:
    keys = ["pred_date", "target_date", "symbol"]
    prediction_keys = predictions_df[keys].copy()
    actuals = actuals_df.copy()
    for key in keys[:2]:
        prediction_keys[key] = pd.to_datetime(prediction_keys[key])
        actuals[key] = pd.to_datetime(actuals[key])

    if prediction_keys.duplicated(keys).any():
        raise ValueError(
            "predictions contain duplicate (pred_date, target_date, symbol) keys"
        )
    if actuals.duplicated(keys).any():
        raise ValueError(
            "actuals contain duplicate (pred_date, target_date, symbol) keys"
        )

    out = prediction_keys.merge(actuals, on=keys, how="left", validate="one_to_one")
    if out["actual_return_pct"].isna().any():
        missing = int(out["actual_return_pct"].isna().sum())
        raise ValueError(f"{missing} prediction row(s) have no matching actual return")

    out["actual_return_pct"] = out["actual_return_pct"].round(4)
    out["actual_magnitude_pct"] = out["actual_magnitude_pct"].round(4)
    out["universe_mean_pct"] = (
        out.groupby("pred_date")["actual_return_pct"].transform("mean").round(4)
    )
    out["pred_date"] = out["pred_date"].dt.strftime("%Y-%m-%d")
    out["target_date"] = out["target_date"].dt.strftime("%Y-%m-%d")
    columns = [
        "pred_date",
        "target_date",
        "symbol",
        "actual_return_pct",
        "actual_direction",
        "actual_magnitude_pct",
        "universe_mean_pct",
    ]
    return out[columns].sort_values(["pred_date", "symbol"]).reset_index(drop=True)
