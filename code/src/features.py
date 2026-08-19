import numpy as np
import pandas as pd
from pathlib import Path

SESSION_OPEN = pd.to_datetime("09:15").time()
SESSION_CLOSE = pd.to_datetime("15:30").time()
LAST_1H_START = pd.to_datetime("14:30").time()
LAST_2H_START = pd.to_datetime("13:30").time()
EOD_WINDOW_START = pd.to_datetime("15:00").time()

TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_SESSION = 375
ANNUALIZATION_MINUTES = TRADING_DAYS_PER_YEAR * MINUTES_PER_SESSION

EWMA_LAMBDA = 0.94
ROLL_20D = 20
ROLL_14D = 14
ROLL_5D = 5

_PARKINSON_CONST = 1.0 / (4.0 * np.log(2.0))


def _prepare_daily(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()
    return df


def _add_trend_momentum_and_label(df: pd.DataFrame) -> pd.DataFrame:
    close_g = df.groupby("symbol")["close"]

    roll_mean_20 = close_g.transform(lambda s: s.rolling(ROLL_20D).mean())
    roll_std_20 = close_g.transform(lambda s: s.rolling(ROLL_20D).std())
    df["price_zscore_20d"] = (df["close"] - roll_mean_20) / roll_std_20

    delta = close_g.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.groupby(df["symbol"]).transform(
        lambda s: s.ewm(alpha=1.0 / ROLL_14D, adjust=False).mean()
    )
    avg_loss = loss.groupby(df["symbol"]).transform(
        lambda s: s.ewm(alpha=1.0 / ROLL_14D, adjust=False).mean()
    )
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)  # all gains -> RSI = 100
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)  # dead flat
    df["rsi_14d"] = rsi

    df["target_date"] = df.groupby("symbol")["date"].shift(-1)
    df["calendar_gap_days"] = (df["target_date"] - df["date"]).dt.days

    next_open = df.groupby("symbol")["open"].shift(-1)
    df["actual_return_pct"] = (next_open / df["close"] - 1.0) * 100.0

    return df


def _add_volatility_and_decay(df: pd.DataFrame) -> pd.DataFrame:
    """vol_1d, vol_1w, vol_1m, vol_ewma_decay, kurtosis_20d, parkinson."""
    ret_g = df.groupby("symbol")["daily_return"]

    df["vol_1d"] = df["daily_return"].abs()
    df["vol_1w"] = ret_g.transform(lambda s: s.rolling(ROLL_5D).std())
    df["vol_1m"] = ret_g.transform(lambda s: s.rolling(ROLL_20D).std())

    sq_ret = df["daily_return"].pow(2)
    ewma_var = sq_ret.groupby(df["symbol"]).transform(
        lambda s: s.ewm(alpha=1.0 - EWMA_LAMBDA, adjust=False).mean()
    )
    df["vol_ewma_decay"] = np.sqrt(ewma_var)

    df["kurtosis_20d"] = ret_g.transform(lambda s: s.rolling(ROLL_20D).kurt())

    df["parkinson_vol_intraday"] = np.sqrt(
        _PARKINSON_CONST * np.log(df["high"] / df["low"]) ** 2
    )
    return df


def _annualized_realized_vol(frame: pd.DataFrame, out_name: str) -> pd.Series:
    """Annualized realized vol from 1-min log returns, vectorized via groupby.agg.

    RV over the window = sum(log_ret^2). We convert to a per-minute variance
    rate (divide by the number of return observations in the window) and
    annualize using the trading-year minute count, rather than assuming a
    fixed number of minutes per window (handles missing/sparse bars safely).
    """
    f = frame.assign(log_ret_sq=frame["log_ret"] ** 2)
    agg = f.groupby(["symbol", "date"]).agg(
        _sq_sum=("log_ret_sq", "sum"), _n=("log_ret", "count")
    )
    per_minute_var = agg["_sq_sum"] / agg["_n"].replace(0, np.nan)
    return np.sqrt(per_minute_var * ANNUALIZATION_MINUTES).rename(out_name)


def build_intraday_features(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-minute bars into per-(date, symbol) intraday features.

    Every row belongs to session T and uses ONLY that session's own minute
    bars up to and including the 15:30 IST close. Rows outside
    [SESSION_OPEN, SESSION_CLOSE] are dropped defensively so that a stray
    T+1 opening-auction print (sometimes timestamped just after the prior
    day in vendor feeds) can never enter a feature.

    Returns a DataFrame indexed by (symbol, date).
    """
    m = minute_df.copy()
    m["timestamp"] = pd.to_datetime(m["timestamp"])
    m["date"] = m["timestamp"].dt.normalize()
    m["time"] = m["timestamp"].dt.time

    m = m[(m["time"] >= SESSION_OPEN) & (m["time"] <= SESSION_CLOSE)]
    m = m.sort_values(["symbol", "date", "timestamp"])

    m["log_ret"] = np.log(m["close"]).groupby([m["symbol"], m["date"]]).diff()

    last_1h = m[m["time"] >= LAST_1H_START]
    last_2h = m[m["time"] >= LAST_2H_START]
    eod_window = m[m["time"] >= EOD_WINDOW_START]

    vol_last_1h = _annualized_realized_vol(last_1h, "vol_last_1h")
    vol_last_2h = _annualized_realized_vol(last_2h, "vol_last_2h")

    last_bar = (
        m.groupby(["symbol", "date"])
        .tail(1)
        .set_index(["symbol", "date"])["close"]
        .rename("_close_1530")
    )
    first_eod_bar = (
        eod_window.sort_values("timestamp")
        .groupby(["symbol", "date"])
        .head(1)
        .set_index(["symbol", "date"])["close"]
        .rename("_close_1500")
    )
    eod_momentum_30m = (last_bar / first_eod_bar - 1.0).rename("eod_momentum_30m")

    eod_volume = eod_window.groupby(["symbol", "date"])["volume"].sum()
    total_volume = m.groupby(["symbol", "date"])["volume"].sum()
    eod_volume_share = (eod_volume / total_volume.replace(0, np.nan)).rename(
        "eod_volume_share"
    )

    intraday = pd.concat(
        [vol_last_1h, vol_last_2h, eod_momentum_30m, eod_volume_share], axis=1
    ).reset_index()

    intraday = intraday.sort_values(["symbol", "date"])
    intraday["intraday_vol_rank"] = intraday.groupby("symbol")["vol_last_1h"].transform(
        lambda s: s.rolling(ROLL_20D).apply(
            lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False
        )
    )
    return intraday


def build_intraday_features_from_path(path: str | Path) -> pd.DataFrame:
    """Build intraday features without materializing the whole minute universe.

    Minute files are independent by symbol. Read one file, reduce it to one
    row per (symbol, date), then release the minute-level frame before reading
    the next file. This keeps peak memory proportional to the largest symbol
    file rather than to the entire minute dataset.
    """
    minute_data_path = Path(path)
    frames = []
    for file_path in sorted(minute_data_path.glob("*.parquet")):
        minute_df = pd.read_parquet(file_path)
        minute_df["symbol"] = file_path.stem
        frames.append(build_intraday_features(minute_df))

    if not frames:
        raise ValueError(f"No Parquet minute files found in {minute_data_path}")
    return pd.concat(frames, ignore_index=True)


def _add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """cs_vol_rank_1h, cs_rel_volume, cs_dispersion_1d.

    Every calculation groups by `pred_date` (session T) across the universe
    of symbols trading on T. This still sits inside F(T): the constraint on
    leakage is the timestamp, not whether a feature is single-name or
    cross-sectional.
    """
    by_date = df.groupby("pred_date")

    df["cs_vol_rank_1h"] = by_date["vol_last_1h"].rank(pct=True)

    vol_mean = by_date["volume"].transform("mean")
    vol_std = by_date["volume"].transform("std")
    df["cs_rel_volume"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    df["cs_dispersion_1d"] = by_date["daily_return"].transform("std")
    return df


FEATURE_COLUMNS = [
    "price_zscore_20d",
    "rsi_14d",
    "calendar_gap_days",
    "vol_1d",
    "vol_1w",
    "vol_1m",
    "vol_ewma_decay",
    "vol_last_1h",
    "vol_last_2h",
    "intraday_vol_rank",
    "eod_momentum_30m",
    "eod_volume_share",
    "parkinson_vol_intraday",
    "cs_vol_rank_1h",
    "cs_rel_volume",
    "cs_dispersion_1d",
    "kurtosis_20d",
]

NON_FEATURE_COLUMNS = ["pred_date", "target_date", "symbol", "actual_return_pct"]


def build_features(
    daily_df: pd.DataFrame, minute_data: pd.DataFrame | str | Path
) -> pd.DataFrame:
    """Build the full leakage-free feature matrix for overnight-return forecasting.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Columns [date, open, high, low, close, volume, symbol].
    minute_data : pd.DataFrame or path
        Either the in-memory minute-bar DataFrame with columns
        [timestamp, open, high, low, close, volume, symbol], or a directory
        containing one Parquet file per symbol. Directory input is processed
        one file at a time to keep peak memory bounded.

    Returns
    -------
    pd.DataFrame
        Indexed by a (`pred_date`, `symbol`) MultiIndex, sorted by
        `pred_date` then `symbol`. Columns are `FEATURE_COLUMNS` plus
        `target_date` and the training label `actual_return_pct`
        (see module docstring — this is a label, not a feature).

        Rows whose rolling windows have not yet warmed up (the first ~20
        sessions per symbol) contain NaNs and are dropped, since they
        cannot be safely used for either training or scoring.
    """
    daily = _prepare_daily(daily_df)
    daily = _add_trend_momentum_and_label(daily)
    daily = _add_volatility_and_decay(daily)

    if isinstance(minute_data, (str, Path)):
        intraday = build_intraday_features_from_path(minute_data)
    else:
        intraday = build_intraday_features(minute_data)

    merged = daily.merge(intraday, on=["symbol", "date"], how="left")
    merged = merged.rename(columns={"date": "pred_date"})

    merged = _add_cross_sectional_features(merged)

    keep = (
        ["pred_date", "symbol"] + FEATURE_COLUMNS + ["target_date", "actual_return_pct"]
    )
    out = merged[keep].copy()
    out = out.dropna(subset=FEATURE_COLUMNS + ["target_date"]).reset_index(drop=True)
    out = out.sort_values(["pred_date", "symbol"]).set_index(["pred_date", "symbol"])
    return out
