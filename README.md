# Machine Learning-Enhanced EMA Trading Strategy in European Equity Markets

**DEDA group project | 2021–2025 daily data | 10 European-country equity ETFs**

This repository contains the source code, supplied input dataset, model outputs, and research figures for our study of a Random Forest confidence filter applied to a traditional EMA(10, 50) crossover strategy. The assets track **US-listed ETFs representing European national equity markets**, not necessarily stocks listed on European exchanges.

## Quick navigation

| Folder / file | Contents |
| --- | --- |
| [`Code/European_ML_EMA_Concise.py`](Code/European_ML_EMA_Concise.py) | Load data; engineer 25 features; train/evaluate Random Forest; select confidence threshold; generate signals; backtest strategies; save core results |
| [`Code/Generate_Research_Figures.py`](Code/Generate_Research_Figures.py) | Read core results; generate 14 research figures; analyze transaction-cost and strict dollar-neutral robustness |
| [`Data/European_ETF_Daily_2021_2025.csv`](Data/European_ETF_Daily_2021_2025.csv) | Supplied daily OHLCV/adjusted-price input (10 ETFs, 2021–2025) |
| [`Results/`](Results/) | 13 CSV results, a run summary, and 14 PNG figures |
| [`SLIDE_CODE_MAP.md`](SLIDE_CODE_MAP.md) | **Mapping from presentation slide numbers to source code and results** |
| [`Presentation/`](Presentation/) | Reference 31-slide outline **and the clean, editable slides 14–26 with speaker notes**; not yet a merged group-final presentation. |
| [`requirements.txt`](requirements.txt) | Python dependencies |

## How to reproduce

In a terminal, change to this repository's root directory and run:

```bash
python3 -m pip install -r requirements.txt
python3 Code/European_ML_EMA_Concise.py
python3 Code/Generate_Research_Figures.py
```

The **order matters**: the figure script reads core CSV outputs created by the model script. Both scripts use paths relative to their own locations. No manual editing of Mac/Windows user paths is needed.

The input ETF CSV is **already provided**. This repository reproduces the results *from that CSV*; it does not include a script to download/verify the original market data source. Training results and plots may differ slightly with library versions/platforms. The threshold selected by the supplied run is 0.59, and some figure annotations refer to that threshold explicitly.

### Experimental design

- Train / feature ranking: 2021–2023; threshold validation: 2024; held-out out-of-sample backtest: 2025.
- Features: 25 technical indicators; Random Forest uses 300 trees (max depth 6; minimum leaf size 12), with top five selected by training-only feature importance.
- Signal: EMA(10/50) supplies long/short direction; RF supplies a direction-aware confidence estimate.
- Filter rules: fixed 0.58; validation-selected 0.59; and ETF-specific dynamic rolling 90th percentile based on past confidence scores.
- Strategies: equal-weight buy-and-hold-style daily-rebalanced benchmark, unfiltered EMA, and three RF-filtered EMA rules.
- Main portfolio: gross exposure normalized to 1 when active, **not** constrained to be dollar neutral.
- Main EMA strategies assume 5 bps one-way transaction cost per unit of turnover; **equal-weight benchmark does not deduct transaction costs**.

### Reading results responsibly

The RF test AUC is close to random (roughly 0.47–0.56). In the supplied 2025 historical backtest, filtered EMA improved several risk measures relative to unfiltered EMA, but **none beat the equal-weight benchmark on total return**. The strict dollar-neutral check leaves the filtered strategies with only 1–2 nonzero-trading days; its near-zero returns must not be interpreted as evidence of a robust, actively deployed neutral strategy.

This is a historical backtest and educational analysis, **not live trading or investment advice**. Data redistribution rights and provenance should be confirmed before publishing the input dataset publicly.

## Summary of results files

| File | Use |
| --- | --- |
| `feature_importance.csv` | Ranked Random Forest features (slide 14) |
| `auc_by_ticker.csv` | 2024 validation and 2025 test AUC by ETF (slide 15) |
| `threshold_tuning_2024.csv` | 2024 threshold-versus-Sharpe evidence (slide 18) |
| `test_2025_predictions_and_signals.csv` | RF probabilities, EMA direction, confidence, filtered signals (slides 16–20) |
| `weights_*.csv` | Portfolio weights by strategy (slides 19–20, 24–25) |
| `portfolio_daily_returns_2025.csv` | 2025 wealth and drawdown paths (slides 21, 23) |
| `performance_summary_2025.csv` | Return, risk, turnover, drawdown and Sharpe (slide 22) |
| `transaction_cost_sensitivity_2025.csv` | 0/5/10/20 bps robustness (slide 24) |
| `strict_dollar_neutral_robustness_2025.csv` | Strict neutrality check including trading days (slide 25) |
| `return_correlation_2021_2025.csv` | ETF return correlations (slide 8) |
| `run_summary.txt` | Short reproducibility summary |

See [`SLIDE_CODE_MAP.md`](SLIDE_CODE_MAP.md) for the full 27-slide mapping and image names.

## Verified reproduction and presentation handoff

The two Python files were executed in an isolated copy of this ZIP. All **13 numerical result CSV files** matched the submitted values within floating point tolerance (maximum absolute discrepancy approximately 7.5e-14), and all **14 figures** regenerated successfully. This verifies computational reproducibility from the included input CSV, **not** the original market-data provenance or all backtest assumptions.

The present repository copy contains a validated reproducibility reference and the updated 13-slide partial deck (slides 14–26). The closing page will be duplicated from the front group’s cover; the merged group-final presentation is **not yet** available. The Q icons and GitHub URL placeholder in the PPT do not have working hyperlinks until the public repository link is inserted.


## Group Members

- Xuanyu Chen
- Wei Li
- Lusi Huang
- Fandi Lyu
- Rui Mo
