from pathlib import Path
import yaml
from src.data import actuals_for_predictions, build_actuals, get_daily_df
from src.evaluate import compute_statistics
from src.features import build_features
from src.models import generate_predictions
from src.utils.backtest import build_daily_profits

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "config.yaml", "r") as file:
    config = yaml.safe_load(file)

train_start = config["splits"]["train"]["start_date"]
train_end = config["splits"]["train"]["end_date"]
valid_end = config["splits"]["valid"]["end_date"]
daily_path = BASE_DIR / config["paths"]["daily_data"]
minute_path = BASE_DIR / config["paths"]["minute_data"]
global_seed = config["seed"]["global"]
weights_path = BASE_DIR / config["model_weights"]["path"]

output_dir = BASE_DIR / config["paths"]["output_dir"]
output_dir.mkdir(parents=True, exist_ok=True)

print("Started")
daily_df = get_daily_df(daily_path)
actuals_df = build_actuals(daily_df)

print("Started building features df")
feature_df = build_features(daily_df, minute_path)
print("Generated features df")
preds_df = generate_predictions(
    feature_df,
    train_end,
    valid_end,
    weights_path=weights_path,
    reuse_existing_weights=config["model_weights"]["reuse_existing"],
)
preds_df.to_csv(output_dir / "predictions.csv", index=False)


actuals_df = actuals_for_predictions(actuals_df, preds_df)
actuals_df.to_csv(output_dir / "actuals.csv", index=False)
print("Generated actuals.csv")

statistics_df = compute_statistics(preds_df, actuals_df)
statistics_df.to_csv(output_dir / "statistics.csv", index=False, float_format="%.6f")

profits_df = build_daily_profits(preds_df, actuals_df, **config["backtest"])
profits_df.to_csv(output_dir / "profits.csv", index=False)
overall_profit = profits_df["cumulative_net_profit"].iloc[-1]
ending_equity = profits_df["equity"].iloc[-1]
print(f"Overall net profit: {overall_profit:,.2f}")
print(f"Ending equity: {ending_equity:,.2f}")
print("Success")
