# Slide-to-Code-and-Result Map (Slides 1–26 + Final Cover to Be Added)

This is an **index**, not 27 independent programs. Run the complete pipeline using the two scripts in `Code/`, in the order shown in the root README. The two original scripts have **not been split or modified** to fit individual slides.

Source code abbreviations:
- **Core**: [`Code/European_ML_EMA_Concise.py`](Code/European_ML_EMA_Concise.py)
- **Figures**: [`Code/Generate_Research_Figures.py`](Code/Generate_Research_Figures.py)
- Chart paths below are relative to `Results/Figures/`; result CSV paths relative to `Results/`.

| Slide | Presentation topic | Source code (functions / logic) | Traceable file(s) |
|---:|---|---|---|
| 1 | Title and research setting | Study metadata | None |
| 2 | Motivation: EMA whipsaw and filters | Conceptual motivation | None |
| 3 | Research questions | Study design, robustness definitions | None |
| 4 | Presentation roadmap | Slide outline | None |
| 5 | Dataset: ten European-country ETFs | Core `main()` reads `Data/European_ETF_Daily_2021_2025.csv` | Input ETF CSV |
| 6 | Data checks and train/validation/test split | Core `main()` data validation and chronological split; forward-label boundary purge | `run_summary.txt`; original data |
| 7 | Market landscape | Figures: normalize 2021–2025 prices | `01_market_normalized_prices.png` |
| 8 | ETF correlation | Figures: correlation matrix and export | `04_return_correlation_heatmap.png`; `return_correlation_2021_2025.csv` |
| 9 | End-to-end workflow | Core `build_panel()`, `fit_rf()`, `add_*signal()`, `portfolio_from_signal()` | Overview of pipeline |
| 10 | 25 technical features | Core `add_features()` and `FEATURES` | `Data/European_ETF_Daily_2021_2025.csv` (input) |
| 11 | Code: feature construction | Core `add_features()` | Same input; example code excerpt |
| 12 | EMA direction | Core `add_features()` (`ema_10`, `ema_50`, `ema_direction`); Figures EMA example | `05_ema_example_EWG.png` |
| 13 | Random Forest specification | Core `fit_rf()`, `main()` (importance ranking, training/validation, refit) | `feature_importance.csv`; `auc_by_ticker.csv` |
| 14 | RF feature importance | Core `fit_rf()`, `main()`; Figures chart #06 | `06_rf_feature_importance.png`; `feature_importance.csv` |
| 15 | 2025 out-of-sample AUC | Core `safe_auc()`, `main()`; Figures chart #07 | `07_test_auc_by_etf.png`; `auc_by_ticker.csv` |
| 16 | Direction-aware confidence (code) | Core `add_direction_confidence()`, `add_static_signal()` | `test_2025_predictions_and_signals.csv` |
| 17 | Three confidence filters | Core `add_static_signal()`, `add_dynamic_signal()`; threshold selection in `main()` | `threshold_tuning_2024.csv`; `test_2025_predictions_and_signals.csv` |
| 18 | 2024 validation threshold search | Core `main()` (`THRESHOLD_GRID`, `tune`); Figures chart #08 | `08_threshold_validation.png`; `threshold_tuning_2024.csv` |
| 19 | Weights, exposure, transaction costs | Core `signal_to_weights()`, `portfolio_from_signal()`, `equal_weight_benchmark()` | `weights_*.csv`; `portfolio_daily_returns_2025.csv` |
| 20 | EWG filtered trading signals (illustration) | Figures chart #12; Core `signal_to_weights()`, `portfolio_from_signal()` | `12_filtered_signals_EWG.png`; `weights_*.csv`; `portfolio_daily_returns_2025.csv` |
| 21 | 2025 cumulative wealth | Figures chart #09; Core `portfolio_from_signal()` | `09_portfolio_cumulative_returns.png`; `portfolio_daily_returns_2025.csv` |
| 22 | Performance summary | Core `performance_stats()`; supplementary Figures chart #11 | `performance_summary_2025.csv`; optional `11_risk_return_bars.png` |
| 23 | Drawdown and 2025 market participation (tuned strategy: 34/249 active days) | Core `maximum_drawdown()`; Figures chart #10; positive gross weight-day count | `10_portfolio_drawdowns.png`; `performance_summary_2025.csv`; `weights_ml_tuned.csv` |
| 24 | Robustness: 0–20 bps costs | Figures: recompute net returns with saved weights (chart #13) | `13_cost_sensitivity.png`; `transaction_cost_sensitivity_2025.csv`; `weights_*.csv` |
| 25 | Robustness: strict dollar neutrality | Figures: simultaneous long/short weights, chart #14 | `14_strict_neutral_robustness.png`; `strict_dollar_neutral_robustness_2025.csv` |
| 26 | Conclusions and limitations on one slide | Synthesizes results, limitations, future work | `performance_summary_2025.csv`; `auc_by_ticker.csv`; `strict_dollar_neutral_robustness_2025.csv` |
| 27 (planned) | Duplicate the completed front cover as closing slide; not included in the current partial PPT | Repository source/data/result index; URL to be added after publication | Root `README.md`; this slide-code map |

## Supplementary slides (in 31-slide reference outline only)

| Page | Source |
|---:|---|
| 28 | `02_etf_2025_returns.png`; `03_annual_return_matrix.png` |
| 29 | `12_filtered_signals_EWG.png` and Core `add_*signal()` (now also featured on main slide 20) |
| 30 | `11_risk_return_bars.png` and `performance_summary_2025.csv` |
| 31 | `14_strict_neutral_robustness.png` and strict neutrality CSV |

## Figure-generation list (all 14 images)

All 14 figures are generated by [`Code/Generate_Research_Figures.py`](Code/Generate_Research_Figures.py), not by separate plotting scripts:

1. `01_market_normalized_prices.png` — normalized ETF prices (2021–2025)
2. `02_etf_2025_returns.png` — individual 2025 ETF returns
3. `03_annual_return_matrix.png` — market-by-year return heatmap
4. `04_return_correlation_heatmap.png` — daily return correlation matrix
5. `05_ema_example_EWG.png` — EWG with EMA(10/50) trend lines
6. `06_rf_feature_importance.png` — Random Forest feature importance
7. `07_test_auc_by_etf.png` — out-of-sample 2025 AUC
8. `08_threshold_validation.png` — 2024 validation Sharpe vs cutoff
9. `09_portfolio_cumulative_returns.png` — strategy wealth curves in 2025
10. `10_portfolio_drawdowns.png` — 2025 drawdown paths
11. `11_risk_return_bars.png` — return, volatility, absolute drawdown comparison
12. `12_filtered_signals_EWG.png` — RF-approved trades for EWG
13. `13_cost_sensitivity.png` — transaction cost robustness
14. `14_strict_neutral_robustness.png` — strict dollar-neutral returns

**Important:** `Presentation/REFERENCE_31_slide_outline_replace_with_final_deck.pptx` is not the group's final merged deck. When the final 27-slide PowerPoint is available, replace or add it under `Presentation/` and update the README accordingly. For slides 14–27, the map above follows the provided 31-slide outline, and is applicable where page numbering/content remains unchanged after the group merge.

## Current handoff (October 2026)

`Presentation/Slides_14-26_Clean_Editable_with_Notes.pptx` is the **current clean, editable 13-slide partial deck (slides 14–26)**. It uses the supplied university 4:3 template, includes speaker notes, and reserves QuantLet-style Q icons as **unlinked placeholders** for a future public GitHub URL.

Slide 26 combines conclusions and limitations. The repeated-cover ending slide has been intentionally omitted pending the front group's actual title slide. The presentation in the repository is **not yet a merged group-final deck**.

Slides 1–13 in the map remain an outline, not independently verified content of the other group's finished slides. After merging, confirm slide numbering, then update this map.
