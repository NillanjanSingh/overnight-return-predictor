import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def get_direction_score(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for split_name, split_df in df.groupby("split"):
        d = split_df["pred_direction"]
        n_obs = len(split_df)
        a_pooled = split_df["actual_return_pct"]
        abs_a_pooled = a_pooled.abs()
        if abs_a_pooled.sum() == 0:
            pooled_score = 0.0
        else:
            pooled_score = (d * a_pooled).sum() / abs_a_pooled.sum()
        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "direction_score",
                "value": pooled_score,
                "n_obs": n_obs,
            }
        )
        a_residual = split_df["actual_return_pct"] - split_df["universe_mean_pct"]
        abs_a_residual = a_residual.abs()
        if abs_a_residual.sum() == 0:
            residual_score = 0.0
        else:
            residual_score = (d * a_residual).sum() / abs_a_residual.sum()
        results.append(
            {
                "split": split_name,
                "scope": "residual",
                "metric": "direction_score",
                "value": residual_score,
                "n_obs": n_obs,
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_directional_return_pct(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for split_name, split_df in df.groupby("split"):
        d = split_df["pred_direction"]
        n_obs = len(split_df)
        a_pooled = split_df["actual_return_pct"]
        pooled_score = (d * a_pooled).mean()
        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "directional_return_pct",
                "value": pooled_score,
                "n_obs": n_obs,
            }
        )
        a_residual = split_df["actual_return_pct"] - split_df["universe_mean_pct"]
        residual_score = (d * a_residual).mean()
        results.append(
            {
                "split": split_name,
                "scope": "residual",
                "metric": "directional_return_pct",
                "value": residual_score,
                "n_obs": n_obs,
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_conf_direction_score(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for split_name, split_df in df.groupby("split"):
        p = split_df["conf_direction"]
        w = 2 * p - 1
        d = split_df["pred_direction"]
        n_obs = len(split_df)
        a_pooled = split_df["actual_return_pct"]
        abs_a_pooled = a_pooled.abs()
        if (w * abs_a_pooled).sum() == 0:
            pooled_score = 0.0
        else:
            pooled_score = (w * d * a_pooled).sum() / (w * abs_a_pooled).sum()
        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "conf_direction_score",
                "value": pooled_score,
                "n_obs": n_obs,
            }
        )
        a_residual = split_df["actual_return_pct"] - split_df["universe_mean_pct"]
        abs_a_residual = a_residual.abs()
        if (w * abs_a_residual).sum() == 0:
            residual_score = 0.0
        else:
            residual_score = (w * d * a_residual).sum() / (w * abs_a_residual).sum()
        results.append(
            {
                "split": split_name,
                "scope": "residual",
                "metric": "conf_direction_score",
                "value": residual_score,
                "n_obs": n_obs,
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_magnitude_score(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for split_name, split_df in df.groupby("split"):
        m = split_df["pred_magnitude_pct"]
        n_obs = len(split_df)
        a_pooled = split_df["actual_return_pct"]
        abs_a_pooled = a_pooled.abs()
        x = (m - abs_a_pooled).abs()
        if abs_a_pooled.sum() == 0:
            pooled_score = 0.0
        else:
            pooled_score = x.sum() / abs_a_pooled.sum()
        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "magnitude_score",
                "value": 1 - pooled_score,
                "n_obs": n_obs,
            }
        )
        a_residual = split_df["actual_return_pct"] - split_df["universe_mean_pct"]
        abs_a_residual = a_residual.abs()
        x = (m - abs_a_residual).abs()
        if abs_a_residual.sum() == 0:
            residual_score = 0.0
        else:
            residual_score = x.sum() / abs_a_residual.sum()
        results.append(
            {
                "split": split_name,
                "scope": "residual",
                "metric": "magnitude_score",
                "value": 1 - residual_score,
                "n_obs": n_obs,
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_conf_direction_lift(df: pd.DataFrame) -> pd.DataFrame:
    base_df = get_direction_score(df)
    conf_df = get_conf_direction_score(df)
    merged = pd.merge(
        conf_df, base_df, on=["split", "scope", "n_obs"], suffixes=("_conf", "_base")
    )
    merged["value"] = merged["value_conf"] - merged["value_base"]
    merged["metric"] = "conf_direction_lift"
    merged = merged[["split", "scope", "metric", "value", "n_obs"]]
    assert type(merged) is pd.DataFrame
    return merged


def get_conf_magnitude_score(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for split_name, split_df in df.groupby("split"):
        c = split_df["conf_magnitude"]
        m = split_df["pred_magnitude_pct"]
        n_obs = len(split_df)
        a_pooled = split_df["actual_return_pct"]
        error_pooled = -(m - a_pooled.abs()).abs()
        corr_pooled, _ = spearmanr(c, error_pooled)
        if pd.isna(corr_pooled):
            corr_pooled = 0.0
        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "conf_magnitude_score",
                "value": corr_pooled,
                "n_obs": n_obs,
            }
        )
        a_residual = split_df["actual_return_pct"] - split_df["universe_mean_pct"]
        error_residual = -(m - a_residual.abs()).abs()
        corr_residual, _ = spearmanr(c, error_residual)
        if pd.isna(corr_residual):
            corr_residual = 0.0
        results.append(
            {
                "split": split_name,
                "scope": "residual",
                "metric": "conf_magnitude_score",
                "value": corr_residual,
                "n_obs": n_obs,
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_hit_rate(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for split_name, split_df in df.groupby("split"):
        n_obs = len(split_df)
        d = split_df["pred_direction"]
        a_pooled = split_df["actual_return_pct"]
        a_sign_pooled = np.where(a_pooled >= 0, 1, -1)
        correct_pooled = (d == a_sign_pooled).astype(int)
        pooled_score = correct_pooled.mean()
        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "hit_rate",
                "value": pooled_score,
                "n_obs": n_obs,
            }
        )
        a_residual = split_df["actual_return_pct"] - split_df["universe_mean_pct"]
        a_sign_residual = np.where(a_residual >= 0, 1, -1)
        correct_residual = (d == a_sign_residual).astype(int)
        residual_score = correct_residual.mean()
        results.append(
            {
                "split": split_name,
                "scope": "residual",
                "metric": "hit_rate",
                "value": residual_score,
                "n_obs": n_obs,
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_precision_up(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for split_name in df["split"].unique():
        split_df = df.loc[(df["split"] == split_name) & (df["pred_direction"] == 1)]
        n_obs = len(split_df)
        temp = split_df.loc[split_df["actual_return_pct"] >= 0]
        pooled_score = len(temp) / n_obs if n_obs != 0 else 0.0
        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "precision_up",
                "value": pooled_score,
                "n_obs": n_obs,
            }
        )
        temp = split_df.loc[
            (split_df["actual_return_pct"] - split_df["universe_mean_pct"]) >= 0
        ]
        residual_score = len(temp) / n_obs if n_obs != 0 else 0.0
        results.append(
            {
                "split": split_name,
                "scope": "residual",
                "metric": "precision_up",
                "value": residual_score,
                "n_obs": n_obs,
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_recall_up(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for split_name, split_df in df.groupby("split"):
        p_df = split_df.loc[split_df["actual_return_pct"] >= 0]
        n_obs = len(p_df)
        temp = p_df.loc[p_df["pred_direction"] == 1]
        pooled_score = len(temp) / n_obs if n_obs != 0 else 0.0
        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "recall_up",
                "value": pooled_score,
                "n_obs": n_obs,
            }
        )
        r_df = split_df.loc[
            (split_df["actual_return_pct"] - split_df["universe_mean_pct"]) >= 0
        ]
        n_obs = len(r_df)
        temp = r_df.loc[r_df["pred_direction"] == 1]
        residual_score = len(temp) / n_obs if n_obs != 0 else 0.0
        results.append(
            {
                "split": split_name,
                "scope": "residual",
                "metric": "recall_up",
                "value": residual_score,
                "n_obs": n_obs,
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_f1_up(df: pd.DataFrame) -> pd.DataFrame:
    prec_df = get_precision_up(df)
    rec_df = get_recall_up(df)
    merged = pd.merge(
        prec_df, rec_df, on=["split", "scope"], suffixes=("_prec", "_rec")
    )
    p = merged["value_prec"]
    r = merged["value_rec"]
    merged["value"] = np.where((p + r) == 0, 0.0, 2 * (p * r) / (p + r))
    merged["metric"] = "f1_up"
    split_sizes = df.groupby("split").size().reset_index(name="n_obs")
    merged = merged.drop(columns=["n_obs_prec", "n_obs_rec"])
    merged = pd.merge(merged, split_sizes, on="split")
    merged = merged[["split", "scope", "metric", "value", "n_obs"]]
    assert type(merged) is pd.DataFrame
    return merged


def get_brier(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for split_name, split_df in df.groupby("split"):
        p = split_df["conf_direction"]
        d = split_df["pred_direction"]
        n_obs = len(split_df)
        a_pooled = split_df["actual_return_pct"]
        correct_pooled = (d == np.where(a_pooled >= 0, 1, -1)).astype(int)
        brier_pooled = ((p - correct_pooled) ** 2).mean()
        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "brier",
                "value": brier_pooled,
                "n_obs": n_obs,
            }
        )
        a_residual = split_df["actual_return_pct"] - split_df["universe_mean_pct"]
        correct_residual = (d == np.where(a_residual >= 0, 1, -1)).astype(int)
        brier_residual = ((p - correct_residual) ** 2).mean()
        results.append(
            {
                "split": split_name,
                "scope": "residual",
                "metric": "brier",
                "value": brier_residual,
                "n_obs": n_obs,
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_brier_skill(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for split_name, split_df in df.groupby("split"):
        p = split_df["conf_direction"]
        d = split_df["pred_direction"]
        n_obs = len(split_df)
        a_pooled = split_df["actual_return_pct"]
        correct_pooled = (d == np.where(a_pooled >= 0, 1, -1)).astype(int)
        brier_pooled = ((p - correct_pooled) ** 2).mean()
        p_ref_pooled = correct_pooled.mean()
        brier_ref_pooled = ((p_ref_pooled - correct_pooled) ** 2).mean()
        skill_pooled = (
            1.0 - (brier_pooled / brier_ref_pooled) if brier_ref_pooled != 0 else 0.0
        )
        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "brier_skill",
                "value": skill_pooled,
                "n_obs": n_obs,
            }
        )
        a_residual = split_df["actual_return_pct"] - split_df["universe_mean_pct"]
        correct_residual = (d == np.where(a_residual >= 0, 1, -1)).astype(int)
        brier_residual = ((p - correct_residual) ** 2).mean()
        p_ref_residual = correct_residual.mean()
        brier_ref_residual = ((p_ref_residual - correct_residual) ** 2).mean()
        skill_residual = (
            1.0 - (brier_residual / brier_ref_residual)
            if brier_ref_residual != 0
            else 0.0
        )
        results.append(
            {
                "split": split_name,
                "scope": "residual",
                "metric": "brier_skill",
                "value": skill_residual,
                "n_obs": n_obs,
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_log_loss(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for split_name, split_df in df.groupby("split"):
        p = np.clip(split_df["conf_direction"], 1e-6, 1.0 - 1e-6)
        d = split_df["pred_direction"]
        n_obs = len(split_df)
        a_pooled = split_df["actual_return_pct"]
        correct_pooled = (d == np.where(a_pooled >= 0, 1, -1)).astype(int)
        log_loss_pooled = -(
            correct_pooled * np.log(p) + (1 - correct_pooled) * np.log(1 - p)
        ).mean()
        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "log_loss",
                "value": log_loss_pooled,
                "n_obs": n_obs,
            }
        )
        a_residual = split_df["actual_return_pct"] - split_df["universe_mean_pct"]
        correct_residual = (d == np.where(a_residual >= 0, 1, -1)).astype(int)
        log_loss_residual = -(
            correct_residual * np.log(p) + (1 - correct_residual) * np.log(1 - p)
        ).mean()
        results.append(
            {
                "split": split_name,
                "scope": "residual",
                "metric": "log_loss",
                "value": log_loss_residual,
                "n_obs": n_obs,
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_ece_10(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    bins = np.linspace(0, 1, 11)
    for split_name, split_df in df.groupby("split"):
        p = split_df["conf_direction"]
        d = split_df["pred_direction"]
        n_obs = len(split_df)

        def calculate_ece_for_scope(correct_array):
            temp_df = pd.DataFrame({"p": p, "correct": correct_array})
            temp_df["bucket"] = pd.cut(temp_df["p"], bins=bins, include_lowest=True)
            ece = 0.0
            for _, group in temp_df.groupby("bucket", observed=False):
                n_b = len(group)
                if n_b > 0:
                    accuracy_b = group["correct"].mean()
                    mean_p_b = group["p"].mean()
                    ece += (n_b / n_obs) * abs(accuracy_b - mean_p_b)
            return ece

        a_pooled = split_df["actual_return_pct"]
        correct_pooled = (d == np.where(a_pooled >= 0, 1, -1)).astype(int)
        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "ece_10",
                "value": calculate_ece_for_scope(correct_pooled),
                "n_obs": n_obs,
            }
        )
        a_residual = split_df["actual_return_pct"] - split_df["universe_mean_pct"]
        correct_residual = (d == np.where(a_residual >= 0, 1, -1)).astype(int)
        results.append(
            {
                "split": split_name,
                "scope": "residual",
                "metric": "ece_10",
                "value": calculate_ece_for_scope(correct_residual),
                "n_obs": n_obs,
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_mae(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for split_name, split_df in df.groupby("split"):
        m = split_df["pred_magnitude_pct"]
        a_pooled = split_df["actual_return_pct"]
        abs_err = (m - a_pooled.abs()).abs()
        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "mae",
                "value": abs_err.mean(),
                "n_obs": len(split_df),
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_rmse(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for split_name, split_df in df.groupby("split"):
        m = split_df["pred_magnitude_pct"]
        a_pooled = split_df["actual_return_pct"]
        sq_err = (m - a_pooled.abs()) ** 2
        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "rmse",
                "value": np.sqrt(sq_err.mean()),
                "n_obs": len(split_df),
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_rank_ic(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for split_name, split_df in df.groupby("split"):
        for scope in ("pooled", "residual"):
            if scope == "pooled":
                a = split_df["actual_return_pct"]
            else:
                a = split_df["actual_return_pct"] - split_df["universe_mean_pct"]

            work = pd.DataFrame(
                {
                    "pred_date": split_df["pred_date"],
                    "symbol": split_df["symbol"],
                    "m": split_df["pred_magnitude_pct"],
                    "abs_a": a.abs(),
                }
            )

            daily_corrs = []
            for _, day_df in work.groupby("pred_date"):
                if day_df["symbol"].nunique() < 2:
                    continue
                corr, _ = spearmanr(day_df["m"], day_df["abs_a"])
                if not pd.isna(corr):
                    daily_corrs.append(corr)

            daily_corrs = np.array(daily_corrs)
            n_days = len(daily_corrs)
            if n_days == 0:
                rank_ic, rank_ic_t = 0.0, 0.0
            else:
                rank_ic = daily_corrs.mean()
                if n_days > 1 and daily_corrs.std(ddof=1) > 0:
                    rank_ic_t = rank_ic / (daily_corrs.std(ddof=1) / np.sqrt(n_days))
                else:
                    rank_ic_t = 0.0
            results.append(
                {
                    "split": split_name,
                    "scope": scope,
                    "metric": "rank_ic",
                    "value": rank_ic,
                    "n_obs": n_days,
                }
            )
            results.append(
                {
                    "split": split_name,
                    "scope": scope,
                    "metric": "rank_ic_t",
                    "value": rank_ic_t,
                    "n_obs": n_days,
                }
            )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_r2_vs_vol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    ordered = df.sort_values(["symbol", "pred_date"]).copy()
    ordered["abs_a_pooled"] = ordered["actual_return_pct"].abs()
    ordered["v"] = (
        ordered.groupby("symbol")["abs_a_pooled"]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=window).mean())
        .reset_index(level=0, drop=True)
    )

    results = []
    for split_name, split_df in ordered.groupby("split"):
        valid = split_df.dropna(subset=["v"])
        m, abs_a, v = valid["pred_magnitude_pct"], valid["abs_a_pooled"], valid["v"]

        denom = ((v - abs_a) ** 2).sum()
        r2 = 1 - ((m - abs_a) ** 2).sum() / denom if denom != 0 and len(valid) else 0.0

        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "r2_vs_vol",
                "value": r2,
                "n_obs": len(valid),
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_mae_conf_deciles(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for split_name, split_df in df.groupby("split"):
        m = split_df["pred_magnitude_pct"]
        a_pooled = split_df["actual_return_pct"]
        abs_err = (m - a_pooled.abs()).abs()
        c = split_df["conf_magnitude"]

        top_mask = c >= c.quantile(0.90)
        bottom_mask = c <= c.quantile(0.10)

        mae_top = abs_err[top_mask].mean() if top_mask.sum() else 0.0
        mae_bottom = abs_err[bottom_mask].mean() if bottom_mask.sum() else 0.0

        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "mae_conf_top_decile",
                "value": mae_top,
                "n_obs": int(top_mask.sum()),
            }
        )
        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "mae_conf_bottom_decile",
                "value": mae_bottom,
                "n_obs": int(bottom_mask.sum()),
            }
        )
        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "conf_mag_gradient",
                "value": mae_bottom - mae_top,
                "n_obs": len(split_df),
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_frac_stocks_hit_gt_50(df: pd.DataFrame, min_days: int = 20) -> pd.DataFrame:
    results = []
    for split_name, split_df in df.groupby("split"):
        d = split_df["pred_direction"]
        a_pooled = split_df["actual_return_pct"]
        correct = (d == np.where(a_pooled >= 0, 1, -1)).astype(int)

        per_symbol = (
            pd.DataFrame({"symbol": split_df["symbol"], "correct": correct})
            .groupby("symbol")
            .agg(hit_rate=("correct", "mean"), n_days=("correct", "size"))
        )
        qualifying = per_symbol[per_symbol["n_days"] >= min_days]
        n_symbols = len(qualifying)
        frac = (qualifying["hit_rate"] > 0.5).mean() if n_symbols else 0.0

        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "frac_stocks_hit_gt_50",
                "value": frac,
                "n_obs": n_symbols,
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_frac_stocks_beat_naive(df: pd.DataFrame, min_days: int = 20) -> pd.DataFrame:
    results = []
    for split_name, split_df in df.groupby("split"):
        d = split_df["pred_direction"]
        a_pooled = split_df["actual_return_pct"]
        correct = (d == np.where(a_pooled >= 0, 1, -1)).astype(int)

        temp = pd.DataFrame(
            {"symbol": split_df["symbol"], "correct": correct, "a": a_pooled.to_numpy()}
        )
        per_symbol = temp.groupby("symbol").agg(
            hit_rate=("correct", "mean"),
            naive_up_rate=("a", lambda s: (s >= 0).mean()),
            n_days=("correct", "size"),
        )
        qualifying = per_symbol[per_symbol["n_days"] >= min_days]
        n_symbols = len(qualifying)
        frac = (
            (qualifying["hit_rate"] > qualifying["naive_up_rate"]).mean()
            if n_symbols
            else 0.0
        )

        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "frac_stocks_beat_naive",
                "value": frac,
                "n_obs": n_symbols,
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def get_var_share_universe(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for split_name, split_df in df.groupby("split"):
        a_pooled = split_df["actual_return_pct"]
        residual = split_df["actual_return_pct"] - split_df["universe_mean_pct"]

        var_a = a_pooled.var(ddof=0)
        var_residual = residual.var(ddof=0)
        share = 1 - var_residual / var_a if var_a != 0 else 0.0

        results.append(
            {
                "split": split_name,
                "scope": "pooled",
                "metric": "var_share_universe",
                "value": share,
                "n_obs": len(split_df),
            }
        )
    stats_df = pd.DataFrame(results)
    stats_df = stats_df[["split", "scope", "metric", "value", "n_obs"]]
    assert type(stats_df) is pd.DataFrame
    return stats_df


def compute_statistics(
    preds_df: pd.DataFrame, actuals_df: pd.DataFrame
) -> pd.DataFrame:
    merge_keys = ["pred_date", "target_date"]
    preds = preds_df.copy()
    actuals = actuals_df.copy()
    for key in merge_keys:
        preds[key] = pd.to_datetime(preds[key])
        actuals[key] = pd.to_datetime(actuals[key])

    df = pd.merge(preds, actuals, on=[*merge_keys, "symbol"])
    frames = [
        get_direction_score(df),
        get_directional_return_pct(df),
        get_conf_direction_score(df),
        get_magnitude_score(df),
        get_conf_direction_lift(df),
        get_conf_magnitude_score(df),
        get_hit_rate(df),
        get_precision_up(df),
        get_recall_up(df),
        get_f1_up(df),
        get_brier(df),
        get_brier_skill(df),
        get_log_loss(df),
        get_ece_10(df),
        get_mae(df),
        get_rmse(df),
        get_rank_ic(df),
        get_r2_vs_vol(df),
        get_mae_conf_deciles(df),
        get_frac_stocks_hit_gt_50(df),
        get_frac_stocks_beat_naive(df),
        get_var_share_universe(df),
    ]
    return pd.concat(frames, ignore_index=True)
