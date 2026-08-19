# Overnight Return Forecasting and Backtesting Pipeline

## Overview
This repository contains a rigorous, leakage-free pipeline for predicting the overnight return of equities (from session close $T$ to the next open $T+1$) and simulating a systematic trading strategy based on those predictions. 

The framework is divided into modular components handling data ingestion, causal feature engineering, multi-faceted predictive modeling (direction, magnitude, and their respective confidences), statistical evaluation, and a full long-only portfolio backtest.

---

## 🚀 Setup & Execution

### Prerequisites
- Python 3.9+
- Dependencies listed in `requirements.txt`

### Directory Structure Requirements
The pipeline expects a specific directory structure for data ingestion, configurable via `config.yaml`. By default, it expects:
```
data/
  ├── daily/      # Daily price bars in .parquet format (one file per symbol)
  └── minute/     # Minute-level price bars in .parquet format (one file per symbol)
Research/
  └── code/       # Source code and main script (this repository)
```

### How to Run
1. **Install dependencies**:
   Navigate to the `code/` directory and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure settings (Optional)**:
   Modify `config.yaml` to adjust hyperparameters, file paths, train/valid/test split dates, embargo days, and backtest configurations.

3. **Execute the pipeline**:
   Run the main script from within the `code/` directory:
   ```bash
   python main.py
   ```

4. **Outputs**:
   Upon successful execution, the pipeline generates the following files in the `output_dir` (default is `../` i.e., the `Research` root):
   - `predictions.csv`: Contains per-symbol predictions (`pred_direction`, `conf_direction`, `pred_magnitude_pct`, `conf_magnitude`).
   - `actuals.csv`: Ground truth data (`actual_return_pct`, `actual_direction`, `actual_magnitude_pct`) matched to predictions.
   - `statistics.csv`: Comprehensive evaluation metrics broken down by split (train/valid/test) and scope (pooled/residual).
   - `profits.csv`: Daily summary of the backtest portfolio's gross/net returns, PnL, and cumulative equity curve.

---

## 🏗️ Project Architecture & Pipeline

The execution flow (`main.py`) integrates several independent modules:
1. **Data Loading (`src/data.py`)**: Ingests raw `.parquet` files for daily and intraday minute resolutions. Aligns dates and calculates the target label `actual_return_pct`.
2. **Feature Engineering (`src/features.py`)**: Computes predictive factors while strictly honoring the temporal boundary of session $T$'s close.
3. **Modeling (`src/models.py`)**: Chronologically splits the dataset (Train/Embargo/Valid/Embargo/Test) and fits models targeting direction and magnitude. Saved model weights are supported for reproducibility.
4. **Evaluation (`src/evaluate.py`)**: Scores predictions against ground truth using financial and statistical metrics.
5. **Backtesting (`src/utils/backtest.py`)**: Simulates a discrete next-open long-only strategy incorporating transaction costs, position sizing, and confidence thresholds.

---

## 🔬 Detailed Research & Methodology

### 1. Feature Engineering (The Golden Rule)
To eliminate **look-ahead bias**, a strict temporal constraint is enforced: Every feature for `pred_date = T` is computed using *only* information observable at or before 15:30 IST on day $T$. 
$T+1$ prices are explicitly excluded from feature generation.

Features are classified into three distinct categories:
- **Trend & Momentum (Daily)**: 20-day rolling z-scores, 14-day exponential RSI.
- **Volatility & Tail Risk (Daily)**: 1-day absolute returns, 1-week/1-month rolling standard deviations, EWMA variance (RiskMetrics $\lambda = 0.94$), 20-day kurtosis, and Parkinson High-Low intraday volatility.
- **Intraday Market Microstructure (Minute)**:
  - Realized volatility derived from 1-minute log returns over the last 1 hour and 2 hours of the trading session.
  - End-of-Day (EoD) momentum: Return from 15:00 to 15:30.
  - Volume concentration: Share of daily volume executed in the last 30 minutes.
  - Volatility rank: 20-day rolling percentile rank of the 1-hour realized volatility.

### 2. Modeling Framework
The system splits the data chronologically (e.g., Train: 2020-2024, Valid: 2024-2025, Test: 2025-2026), separated by a **5-day embargo** to prevent auto-correlated leakage between splits. Scaling (StandardScaler) is strictly fit on the training fold.

Four distinct targets are predicted using specialized white-box models:
1. **Direction Model (`pred_direction`)**: An Ordinary Least Squares (Linear Regression) model trained on all features to predict the raw overnight return. The output's sign yields the binary direction prediction (+1 or -1).
2. **Direction Confidence (`conf_direction`)**: A Logistic Regression model ($C=0.1$) trained to predict the probability of a positive return ($>= 0$). Provides calibrated confidence scores.
3. **Magnitude Model (`pred_magnitude_pct`)**: Uses an EWMA filter ($\lambda=0.94$) applied directly to the historical realized overnight return series ($r_{T-1}, r_{T-2}, ...$) to estimate the expected magnitude $|r_T|$.
4. **Magnitude Confidence (`conf_magnitude`)**: Formulated as a structural tail-risk metric. It uses 20-day Fisher excess kurtosis ($\kappa$) and converts it to a normalized confidence scale using $1 - (\kappa - 3) / \max(\kappa_{train})$. Higher kurtosis indicates fat tails (less predictability), lowering the confidence score.

### 3. Evaluation Metrics
Predictions are analyzed across pooled and residual components (returns adjusted against the daily universe mean):
- **Direction Score**: Precision-weighted proxy; $\sum (direction \times return) / \sum |return|$.
- **Confidence Direction Score**: The above direction score, but weighted by the magnitude of confidence $(2p - 1)$.
- **Directional Return Pct**: Mean realized return for the directional predictions.
- **Magnitude Score**: Inverse of the normalized absolute error between predicted magnitude and actual magnitude.
- **Confidence Magnitude Score**: Spearman rank correlation between the magnitude confidence and the negative magnitude prediction error.

### 4. Backtesting Strategy
A realistic, transaction-cost-aware portfolio simulation is executed over the test set:
- **Rules**: Long-only overnight holding. Buys at the close of $T$ and sells at the open of $T+1$.
- **Filtering**: Only considers assets with a positive direction signal (`pred_direction == 1`), high confidence (`conf_direction > 0.8`), and sufficient predicted alpha (`pred_magnitude_pct > 0.5%`).
- **Ranking & Selection**: Sorts eligible candidates primarily by `conf_direction`, followed by `pred_magnitude_pct`. Selects the top `max_positions` (default 25) assets daily.
- **Costs**: Deducts a flat `round_trip_cost_pct` (e.g., 0.15% or 15 bps) per taken position to account for slippage and commissions.
- **Accounting**: Tracks a dynamically updated equity curve (starting at $1,000,000 capital) scaled by equal-weighted net returns, calculating daily net profit and cumulative performance.

---

## 📊 Backtesting Results

Running the portfolio over the out-of-sample **Test** split (May 2025 – June 2026) yielded the following top-line metrics based on a starting capital of **$1,000,000**:

| Metric | Result |
| :--- | :--- |
| **Initial Capital** | 1,000,000.00 |
| **Ending Equity** | 2,078,084.46 |
| **Cumulative Net Profit** | 1,078,084.46 |
| **Net Return** | ~ 107% |

### Detailed Evaluation (Pooled vs Residual)
- **Pooled Hit Rate**: The model successfully identified the overarching market drift, achieving a strong pooled hit rate of **66.67%** on the test set.
- **Residual Degradation**: When the universe mean is subtracted (to isolate idiosyncratic stock picking), the residual hit rate on the test set falls to **49.88%**, effectively a random walk. This indicates much of the predictive edge originates from common market-level overnight behavior.
- **Confidence Calibration**: The `conf_direction_lift` is positive across all splits, reaching **+0.033** on the test pooled scope, proving that higher-conviction predictions correspond to a higher likelihood of being correct.
- **Magnitude Ranking**: A positive Spearman rank correlation of **0.133** on the test set confirms the efficacy of the EWMA magnitude estimator in ranking which stocks will experience the largest absolute gaps.
- **Transaction Friction**: The gross edge is substantial, but profitability is exceptionally sensitive to the assumed **0.15% round-trip transaction cost** due to the high-turnover nature of gap trading.
