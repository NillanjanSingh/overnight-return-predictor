"""Long-only, next-open backtest for the prediction output."""

from collections.abc import Collection
import pandas as pd

REQUIRED_PREDICTION_COLUMNS = {
    "pred_date",
    "target_date",
    "symbol",
    "pred_magnitude_pct",
    "pred_direction",
    "conf_direction",
    "split",
}
REQUIRED_ACTUAL_COLUMNS = {
    "pred_date",
    "target_date",
    "symbol",
    "actual_return_pct",
}
KEYS = ["pred_date", "target_date", "symbol"]


def build_daily_profits(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    *,
    min_conf_direction: float = 0.8,
    min_predicted_return_pct: float = 0.5,
    max_positions: int = 25,
    round_trip_cost_pct: float = 0.15,
    initial_capital: float = 1_000_000.0,
    splits: Collection[str] = ("test",),
) -> pd.DataFrame:
    """Backtest equal-weighted long positions from each session close to next open.

    ``predictions.csv`` does not expose a signed predicted-return column.
    For a +1 direction call, ``pred_magnitude_pct`` is therefore used as the
    predicted positive-return proxy. Eligible names are ranked by direction
    confidence and the highest-confidence ``max_positions`` are bought at the
    close on ``pred_date`` and sold at the open on ``target_date``.

    ``round_trip_cost_pct`` is deducted once per selected position. It covers
    both entry and exit costs, expressed in percentage points of capital.
    By default, only the out-of-sample ``test`` split is traded.
    """
    missing_predictions = REQUIRED_PREDICTION_COLUMNS - set(predictions_df.columns)
    missing_actuals = REQUIRED_ACTUAL_COLUMNS - set(actuals_df.columns)
    if missing_predictions:
        raise ValueError(f"predictions missing columns: {sorted(missing_predictions)}")
    if missing_actuals:
        raise ValueError(f"actuals missing columns: {sorted(missing_actuals)}")
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if round_trip_cost_pct < 0:
        raise ValueError("round_trip_cost_pct cannot be negative")
    if not splits:
        raise ValueError("at least one split must be selected")

    predictions = predictions_df.copy()
    actuals = actuals_df.copy()
    for key in KEYS[:2]:
        predictions[key] = pd.to_datetime(predictions[key])
        actuals[key] = pd.to_datetime(actuals[key])
    predictions = predictions.loc[predictions["split"].isin(splits)].copy()
    if predictions.empty:
        raise ValueError(f"no predictions found for requested splits: {sorted(splits)}")

    trades = predictions.merge(
        actuals[KEYS + ["actual_return_pct"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    if len(trades) != len(predictions):
        raise ValueError(
            "every prediction must have a matching actual before backtesting"
        )

    eligible = trades.loc[
        (trades["pred_direction"] == 1)
        & (trades["conf_direction"] > min_conf_direction)
        & (trades["pred_magnitude_pct"] > min_predicted_return_pct)
    ].copy()
    selected = (
        eligible.sort_values(
            ["pred_date", "conf_direction", "pred_magnitude_pct", "symbol"],
            ascending=[True, False, False, True],
        )
        .groupby("pred_date", group_keys=False)
        .head(max_positions)
    )

    all_days = (
        predictions[["pred_date", "target_date", "split"]]
        .drop_duplicates("pred_date")
        .sort_values("pred_date")
    )
    daily = (
        selected.groupby("pred_date", as_index=False)
        .agg(
            n_positions=("symbol", "size"),
            gross_return_pct=("actual_return_pct", "mean"),
            mean_conf_direction=("conf_direction", "mean"),
        )
        .merge(all_days, on="pred_date", how="right", validate="one_to_one")
        .sort_values("pred_date")
        .reset_index(drop=True)
    )
    daily["n_positions"] = daily["n_positions"].fillna(0).astype(int)
    for column in ["gross_return_pct", "mean_conf_direction"]:
        daily[column] = daily[column].fillna(0.0)
    daily["transaction_cost_pct"] = (
        daily["n_positions"].gt(0).astype(float) * round_trip_cost_pct
    )
    daily["net_return_pct"] = daily["gross_return_pct"] - daily["transaction_cost_pct"]

    equity = initial_capital
    daily_profits = []
    equity_values = []
    for net_return_pct in daily["net_return_pct"]:
        profit = equity * net_return_pct / 100.0
        equity += profit
        daily_profits.append(profit)
        equity_values.append(equity)
    daily["daily_net_profit"] = daily_profits
    daily["equity"] = equity_values
    daily["cumulative_net_profit"] = daily["equity"] - initial_capital

    daily["pred_date"] = daily["pred_date"].dt.strftime("%Y-%m-%d")
    daily["target_date"] = daily["target_date"].dt.strftime("%Y-%m-%d")
    return daily[
        [
            "pred_date",
            "target_date",
            "split",
            "n_positions",
            "mean_conf_direction",
            "gross_return_pct",
            "transaction_cost_pct",
            "net_return_pct",
            "daily_net_profit",
            "cumulative_net_profit",
            "equity",
        ]
    ].round(
        {
            "mean_conf_direction": 6,
            "gross_return_pct": 6,
            "transaction_cost_pct": 6,
            "net_return_pct": 6,
            "daily_net_profit": 2,
            "cumulative_net_profit": 2,
            "equity": 2,
        }
    )
