# Machine Learning-Enhanced EMA Trading Strategy in European Equity Markets

**DEDA Group Project | School of Economics, Fudan University | 2026**  
**Data:** Daily observations for 10 US-listed European-country equity ETFs, 2021–2025

## Group Members

Xuanyu Chen · Wei Li · Lusi Huang · Fandi Lyu · Rui Mo

## Project Overview

This project investigates whether a **Random Forest (RF) confidence filter** can improve a traditional **EMA(10, 50) crossover trading strategy** in European equity markets. We construct technical indicators from daily ETF market data, estimate the probability of positive next-day returns, and use that probability to filter EMA-generated trading signals. We compare the resulting strategies with an unfiltered EMA strategy and an equal-weight ETF benchmark.

The sample consists of **10 US-listed ETFs tracking European national equity markets**. Their USD-denominated returns should not be interpreted as local-currency returns on the underlying European indices.

## Methodology

1. **Data and features:** Process daily price and volume data and construct 25 technical indicators covering momentum, mean reversion, trends, volatility, and trading activity.
2. **Model training:** Train Random Forest classifiers and select the five most important features using the 2021–2023 training data.
3. **Validation:** Use 2024 data to choose a confidence threshold without tuning on the test sample.
4. **Trading signals:** Use EMA(10, 50) to determine long/short direction and RF probabilities to filter signals. Compare a fixed **0.58** threshold, a validation-selected **0.59** threshold, and a dynamic rolling **90th-percentile** threshold.
5. **Backtesting:** Evaluate the strategies on the held-out **2025** sample using return and risk metrics, including transaction-cost sensitivity and a strict dollar-neutral robustness check.

## Out-of-Sample Results (2025)

| Strategy | Total Return | Sharpe Ratio | Maximum Drawdown |
| --- | ---: | ---: | ---: |
| Equal-weight ETF benchmark | 45.12% | 2.31 | -14.31% |
| Traditional EMA | 1.62% | 0.19 | -18.18% |
| RF-filtered EMA: fixed 0.58 | 18.46% | 1.69 | -2.36% |
| RF-filtered EMA: tuned 0.59 | 18.44% | 1.72 | -1.99% |
| RF-filtered EMA: dynamic Q90 | 25.56% | 1.77 | -6.41% |

The RF-filtered strategies improved several risk-adjusted metrics relative to traditional EMA, but **did not outperform the equal-weight benchmark in total return**. Predictive discrimination remained limited: ETF-level test AUC values were approximately **0.47–0.56**.

## Repository Structure

| Path | Description |
| --- | --- |
| `Presentation/` | Project presentation |
| `Code/European_ML_EMA_Concise.py` | Feature engineering, RF modeling, signal construction, and backtesting |
| `Code/Generate_Research_Figures.py` | Research figures and additional robustness analysis |
| `Data/European_ETF_Daily_2021_2025.csv` | Input daily ETF dataset |
| `Results/` | Model outputs, strategy performance tables, and daily portfolio results |
| `Results/Figures/` | 14 research figures |
| `requirements.txt` | Python dependencies |

## Reproducibility

From the repository root, run:

```bash
python3 -m pip install -r requirements.txt
python3 Code/European_ML_EMA_Concise.py
python3 Code/Generate_Research_Figures.py
```

Run the scripts **in this order**: the visualization script uses outputs produced by the main analysis script. Data paths are relative to the repository structure; no local machine-specific paths are required.

