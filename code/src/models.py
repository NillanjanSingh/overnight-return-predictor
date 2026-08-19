import sys
import time
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import threadpoolctl
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

EMBARGO_DAYS = 5
EWMA_LAMBDA = 0.94  # same RiskMetrics decay used in features.py
GAUSSIAN_ABS_CONST = np.sqrt(2.0 / np.pi)
DEFAULT_N_THREADS = 1  # BLAS thread cap; see module docstring

_NON_FEATURE_COLS = {"pred_date", "symbol", "target_date", "actual_return_pct"}


# ----------------------------------------------------------------------------
# Progress instrumentation
# ----------------------------------------------------------------------------
_T_START = None


def _progress(msg: str, verbose: bool = True) -> None:
    """Print `[+elapsed s] msg` and flush immediately.

    Flushing matters here specifically: without it, output can sit in a
    buffer and never appear before a slow/stuck step, which is exactly
    the "is it stuck or just slow?" ambiguity this function exists to kill.
    """
    global _T_START
    if _T_START is None:
        _T_START = time.time()
    if verbose:
        print(f"[models.py +{time.time() - _T_START:6.1f}s] {msg}", flush=True)
        sys.stdout.flush()


def _check_finite(X: np.ndarray, feature_cols: List[str], where: str) -> None:
    """Fail fast and loud on NaN/Inf instead of feeding it to a solver.

    A stray Inf (e.g. a symbol with a zero low-price tick corrupting
    `parkinson_vol_intraday` via log(High/0)) doesn't necessarily crash
    sklearn outright -- depending on version/solver it can instead cause
    the optimizer to wander for a very long time before erroring or
    "converging" to nonsense. This turns that into an immediate, specific
    error instead of an ambiguous multi-minute stall.
    """
    finite_mask = np.isfinite(X)
    if finite_mask.all():
        return
    bad_cols = [
        feature_cols[j] for j in range(X.shape[1]) if not finite_mask[:, j].all()
    ]
    n_bad = int((~finite_mask).sum())
    raise ValueError(
        f"generate_predictions: {n_bad} non-finite value(s) in X at '{where}' "
        f"in column(s) {bad_cols}. Fix upstream in features.py rather than "
        f"letting these reach the solver -- they are a likely cause of "
        f"apparent hangs, not just wrong numbers."
    )


# ----------------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------------
def _assign_splits(
    dates: pd.Series,
    train_end_date: str,
    valid_end_date: str,
    embargo_days: int = EMBARGO_DAYS,
) -> pd.Series:
    """Chronological train / embargo / valid / embargo / test assignment.

    Embargo rows are labelled 'embargo' and dropped from the final output;
    they are never fit on, never scored, never predicted for.
    """
    train_end = pd.Timestamp(train_end_date)
    valid_end = pd.Timestamp(valid_end_date)

    uniq = np.sort(dates.unique())
    after_train = uniq[uniq > train_end]
    after_embargo1 = after_train[embargo_days:]

    valid_dates = set(after_embargo1[after_embargo1 <= valid_end])
    after_valid = after_embargo1[after_embargo1 > valid_end]
    test_dates = set(after_valid[embargo_days:])

    # Vectorized (.isin over the whole Series in one C-level pass) rather
    # than a per-row Python function call via .map() -- matters once N is
    # in the hundreds of thousands. Default is 'embargo'; train/valid/test
    # get explicitly overwritten, so anything unclassified stays excluded.
    result = pd.Series("embargo", index=dates.index, dtype="object")
    result[dates.isin(valid_dates)] = "valid"
    result[dates.isin(test_dates)] = "test"
    result[dates <= train_end] = "train"
    return result


def _feature_columns(df: pd.DataFrame) -> List[str]:
    """All columns that are genuine model inputs (excludes keys + label)."""
    return [c for c in df.columns if c not in _NON_FEATURE_COLS and c != "split"]


def _weight_metadata(
    feature_cols: List[str],
    train_end_date: str,
    valid_end_date: str,
    embargo_days: int,
    random_state: int,
) -> dict:
    return {
        "feature_cols": feature_cols,
        "train_end_date": train_end_date,
        "valid_end_date": valid_end_date,
        "embargo_days": embargo_days,
        "random_state": random_state,
    }


def _load_model_weights(weights_path: Path, metadata: dict) -> dict:
    """Load persisted train-fitted objects after validating their contract."""
    bundle = joblib.load(weights_path)
    required = {"metadata", "scaler", "ols", "logit"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(
            f"model weights at {weights_path} are missing: {sorted(missing)}"
        )
    if bundle["metadata"] != metadata:
        raise ValueError(
            f"model weights at {weights_path} do not match the current feature "
            "schema or split configuration. Set reuse_existing: false to retrain."
        )
    return bundle


def _save_model_weights(
    weights_path: Path,
    metadata: dict,
    scaler: StandardScaler,
    ols: LinearRegression,
    logit: LogisticRegression,
) -> None:
    """Save every train-fitted object used by the direction models."""
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"metadata": metadata, "scaler": scaler, "ols": ols, "logit": logit},
        weights_path,
    )


# ----------------------------------------------------------------------------
# Model 1 & 2: OLS direction, Logistic direction-confidence
# ----------------------------------------------------------------------------
def _fit_direction_models(
    X_train: np.ndarray,
    y_return_train: np.ndarray,
    random_state: int,
    n_threads: int,
    verbose: bool,
) -> tuple[LinearRegression, LogisticRegression]:
    """OLS on raw overnight return; C=0.1 logistic on its sign."""
    with threadpoolctl.threadpool_limits(limits=n_threads):
        _progress(f"fitting OLS direction model on {X_train.shape[0]} rows...", verbose)
        ols = LinearRegression()
        ols.fit(X_train, y_return_train)
        _progress("OLS fit done.", verbose)

        y_up = (y_return_train >= 0).astype(int)
        _progress(
            f"fitting logistic direction-confidence model "
            f"(class balance: {y_up.mean():.3f} up)...",
            verbose,
        )
        logit = LogisticRegression(C=0.1, max_iter=1000, random_state=random_state)
        logit.fit(X_train, y_up)
        _progress(f"logistic fit done ({logit.n_iter_[0]} iterations).", verbose)
    return ols, logit


# ----------------------------------------------------------------------------
# Model 3: EWMA volatility -> expected |overnight return|
# ----------------------------------------------------------------------------
def _ewma_overnight_magnitude(df: pd.DataFrame) -> pd.Series:
    """sigma_{i,T} via EWMA of *lagged, realised* overnight-return variance.

    Design note: the assignment's target for this model is the overnight
    move |r(i,T)|, the SAME quantity the direction OLS targets -- not the
    full close-to-close daily return that `vol_ewma_decay` (features.py)
    tracks. So rather than reusing `vol_ewma_decay`, this model fits its
    own EWMA filter directly on the historical overnight-return series.

    At session T we use r(i, T-1), r(i, T-2), ... -- i.e. `actual_return_pct`
    shifted by one row per symbol. r(i, T-1) = (Open_T / Close_{T-1} - 1)*100
    is fully realised at T's open, long before T's close, so this is
    strictly inside F(T): no look-ahead.

    This is a fixed-decay recursive filter (lambda=0.94 is a hyperparameter,
    not a fitted-on-train statistic), so -- exactly like the rolling
    features in features.py -- it is computed once, causally, across the
    whole timeline rather than being refit per split.
    """
    df = df.sort_values(["symbol", "pred_date"])
    r_lag = df.groupby("symbol")["actual_return_pct"].shift(1)
    sq_lag = r_lag.pow(2)
    ewma_var = sq_lag.groupby(df["symbol"]).transform(
        lambda s: s.ewm(alpha=1.0 - EWMA_LAMBDA, adjust=False).mean()
    )
    sigma = np.sqrt(ewma_var)
    return (sigma * GAUSSIAN_ABS_CONST).rename("pred_magnitude_pct")


# ----------------------------------------------------------------------------
# Model 4: kurtosis-based tail-risk confidence
# ----------------------------------------------------------------------------
def _magnitude_confidence(df: pd.DataFrame) -> pd.Series:
    """1 - (kappa - 3) / max(kappa_train), rescaled to (0.05, 0.95) on train.

    `kurtosis_20d` (features.py) is pandas' Fisher *excess* kurtosis
    (normal distribution -> 0), while the assignment's formula is written
    in the Pearson convention (normal distribution -> kappa = 3). We add
    3 back to restore that convention before applying the formula.
    """
    kappa = df["kurtosis_20d"] + 3.0

    train_mask = df["split"] == "train"
    kappa_train_max = kappa[train_mask].max()
    if not np.isfinite(kappa_train_max) or kappa_train_max == 0:
        kappa_train_max = 1.0  # degenerate-train defensive fallback

    raw_conf = 1.0 - (kappa - 3.0) / kappa_train_max

    scaler = MinMaxScaler(feature_range=(0.05, 0.95))
    scaler.fit(raw_conf[train_mask].to_numpy().reshape(-1, 1))
    scaled = scaler.transform(raw_conf.to_numpy().reshape(-1, 1)).ravel()

    # Defensive clip: MinMaxScaler.transform() does not itself guarantee
    # bounds for valid/test rows whose raw value falls outside the train
    # min/max, so we clip to [0, 1] to honour the "strictly within [0, 1]"
    # requirement even under distribution shift.
    scaled = np.clip(scaled, 1e-6, 1 - 1e-6)
    return pd.Series(scaled, index=df.index, name="conf_magnitude")


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def generate_predictions(
    feature_df: pd.DataFrame,
    train_end_date: str,
    valid_end_date: str,
    embargo_days: int = EMBARGO_DAYS,
    random_state: int = 42,
    n_threads: int = DEFAULT_N_THREADS,
    weights_path: str | Path | None = None,
    reuse_existing_weights: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit the four white-box estimators and produce formatted predictions.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Output of `features.build_features` -- indexed by
        (`pred_date`, `symbol`), with feature columns plus `target_date`
        and the label `actual_return_pct`.
    train_end_date, valid_end_date : str
        Chronological split boundaries (inclusive on the upper end of
        train and valid respectively).
    embargo_days : int, default 5
        Trading days skipped after each boundary; embargoed rows are
        dropped from the output.
    random_state : int, default 42
        Seed for the logistic-regression solver, for reproducibility.
    n_threads : int, default 1
        Caps BLAS threads (via threadpoolctl) during every sklearn
        fit/predict call. See the module docstring -- this is the fix for
        the "generate_predictions freezes the whole machine" symptom on
        multi-core machines. Raise it (e.g. to 4) once you've confirmed
        the hang is gone and want more throughput.
    verbose : bool, default True
        Print a timestamped progress line at each pipeline stage.

    Returns
    -------
    pd.DataFrame
        Columns, in order:
        ['pred_date', 'target_date', 'symbol', 'pred_magnitude_pct',
         'pred_direction', 'conf_direction', 'conf_magnitude', 'split'],
        sorted by `pred_date` then `symbol`. `split` in {'train','valid','test'}
        only -- embargo rows never appear here.
    """
    global _T_START
    _T_START = time.time()

    _progress(f"received feature_df with {len(feature_df)} rows.", verbose)
    df = feature_df.reset_index().copy()
    df["pred_date"] = pd.to_datetime(df["pred_date"])
    df["target_date"] = pd.to_datetime(df["target_date"])

    _progress("assigning train/embargo/valid/embargo/test splits...", verbose)
    df["split"] = _assign_splits(
        df["pred_date"], train_end_date, valid_end_date, embargo_days
    )
    df = df[df["split"] != "embargo"].reset_index(drop=True)
    _progress(
        f"split sizes -> {df['split'].value_counts().to_dict()} "
        f"({len(df)} rows after dropping embargo).",
        verbose,
    )

    feature_cols = _feature_columns(df)
    train_mask = df["split"] == "train"

    # --- Models 1 & 2: fit on train only, predict for every remaining row.
    labelled_train = train_mask & df["actual_return_pct"].notna()
    resolved_weights_path = Path(weights_path) if weights_path is not None else None
    metadata = _weight_metadata(
        feature_cols, train_end_date, valid_end_date, embargo_days, random_state
    )
    bundle = None
    if (
        reuse_existing_weights
        and resolved_weights_path is not None
        and resolved_weights_path.exists()
    ):
        bundle = _load_model_weights(resolved_weights_path, metadata)
        _progress(f"loaded model weights from {resolved_weights_path}.", verbose)

    _progress(
        f"scaling {len(feature_cols)} features "
        f"({int(labelled_train.sum())} labelled train rows)...",
        verbose,
    )
    with threadpoolctl.threadpool_limits(limits=n_threads):
        if bundle is None:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(df.loc[labelled_train, feature_cols])
        else:
            scaler = bundle["scaler"]
            X_train_scaled = scaler.transform(df.loc[labelled_train, feature_cols])
        X_all_scaled = scaler.transform(df[feature_cols])
    _progress("scaling done.", verbose)

    _check_finite(X_train_scaled, feature_cols, "X_train_scaled")
    _check_finite(X_all_scaled, feature_cols, "X_all_scaled")

    if bundle is None:
        ols, logit = _fit_direction_models(
            X_train_scaled,
            df.loc[labelled_train, "actual_return_pct"].to_numpy(),
            random_state=random_state,
            n_threads=n_threads,
            verbose=verbose,
        )
        if resolved_weights_path is not None:
            _save_model_weights(resolved_weights_path, metadata, scaler, ols, logit)
            _progress(f"saved model weights to {resolved_weights_path}.", verbose)
    else:
        ols, logit = bundle["ols"], bundle["logit"]

    _progress("scoring direction models on all rows...", verbose)
    with threadpoolctl.threadpool_limits(limits=n_threads):
        raw_return_hat = ols.predict(X_all_scaled)
        proba_up = logit.predict_proba(X_all_scaled)[:, 1]
    pred_direction = np.sign(raw_return_hat)
    pred_direction[pred_direction == 0] = 1.0
    df["pred_direction"] = pred_direction.astype(int)
    df["conf_direction"] = np.clip(proba_up, 1e-6, 1 - 1e-6)
    _progress("direction scoring done.", verbose)

    # --- Model 3: EWMA overnight-volatility -> expected |return|.
    _progress("computing EWMA overnight-magnitude filter...", verbose)
    df["pred_magnitude_pct"] = _ewma_overnight_magnitude(df).clip(lower=0.0)
    _progress("magnitude filter done.", verbose)

    # --- Model 4: kurtosis-based tail-risk confidence.
    _progress("computing kurtosis-based magnitude confidence...", verbose)
    df["conf_magnitude"] = _magnitude_confidence(df)
    _progress("magnitude confidence done.", verbose)

    # Rows whose model-3 EWMA filter has not warmed up yet (each symbol's
    # very first available session) cannot be safely scored -- drop them,
    # per the "dropping missing initial window rows" guidance.
    n_before = len(df)
    df = df.dropna(
        subset=[
            "pred_magnitude_pct",
            "conf_magnitude",
            "conf_direction",
            "pred_direction",
        ]
    )
    if len(df) != n_before:
        _progress(
            f"dropped {n_before - len(df)} row(s) with unwarmed-up "
            f"magnitude filters (each symbol's first session).",
            verbose,
        )

    df["pred_magnitude_pct"] = df["pred_magnitude_pct"].round(4)

    out_cols = [
        "pred_date",
        "target_date",
        "symbol",
        "pred_magnitude_pct",
        "pred_direction",
        "conf_direction",
        "conf_magnitude",
        "split",
    ]
    out = df[out_cols].sort_values(["pred_date", "symbol"]).reset_index(drop=True)
    _progress(f"done. returning {len(out)} rows.", verbose)
    return out
