# GARCH & Volatility Forecasting

Python-based volatility forecasting framework using GARCH(1,1),
EGARCH, GJR-GARCH, and HAR-RV models with rolling estimation,
regime classification, and predictive accuracy evaluation
across multi-asset portfolios.

---

## Overview

Volatility forecasting is central to market risk management —
from VaR estimation to options pricing and portfolio
construction. This project builds a comprehensive suite of
volatility models, evaluates short-horizon predictive accuracy,
and tests robustness across market regimes including crisis
and low-volatility periods.

---

## Models Implemented

**GARCH(1,1)**
- Maximum likelihood estimation
- Conditional variance dynamics
- Persistence and mean reversion analysis
- 22-day ahead volatility forecasting

**EGARCH**
- Asymmetric volatility response (leverage effect)
- Log-variance specification
- News impact curve analysis
- Comparison with symmetric GARCH

**GJR-GARCH**
- Threshold effects for negative returns
- Asymmetry coefficient estimation
- AIC/BIC model comparison

**HAR-RV (Heterogeneous Autoregressive Realized Variance)**
- Daily, weekly, monthly realized variance components
- Long-memory volatility properties
- Corsi (2009) methodology

**EWMA (RiskMetrics)**
- Lambda decay factor (0.94 standard)
- Benchmark comparison model

---

## Rolling Estimation and Regime Analysis

- Rolling windows: 21-day, 63-day, 126-day, 252-day
- Volatility regime classification: Low, Medium, High
- EWMA vs rolling volatility comparison
- Structural break and clustering analysis

---

## Forecast Evaluation

- MSE and QLIKE loss functions
- Diebold-Mariano test for forecast comparison
- Walk-forward out-of-sample validation
- Realized variance as benchmark proxy

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-%233670A0.svg?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/NumPy-%230288D1.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-%234527A0.svg?style=for-the-badge&logo=pandas&logoColor=white) ![Statsmodels](https://img.shields.io/badge/Statsmodels-%2300BFA5.svg?style=for-the-badge&logo=python&logoColor=white) ![Scipy](https://img.shields.io/badge/SciPy-%231565C0.svg?style=for-the-badge&logo=scipy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23C62828.svg?style=for-the-badge&logo=Matplotlib&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-%2300C853.svg?style=for-the-badge&logo=plotly&logoColor=white)

---

## Project Structure

```
GARCH-Volatility-Forecasting/
│
├── data/
│   ├── returns.csv
│   └── prices.csv
│
├── notebooks/
│   ├── 01_garch_estimation.ipynb
│   ├── 02_egarch_gjr_garch.ipynb
│   ├── 03_har_rv_model.ipynb
│   ├── 04_rolling_estimation.ipynb
│   └── 05_forecast_evaluation.ipynb
│
├── src/
│   ├── garch_models.py
│   ├── har_rv.py
│   ├── rolling_vol.py
│   └── forecast_evaluation.py
│
├── results/
│   ├── garch11_conditional_vol.png
│   ├── egarch_gjr_comparison.png
│   ├── news_impact_curves.png
│   ├── har_rv_fit.png
│   ├── rolling_volatility_windows.png
│   ├── volatility_regimes.png
│   ├── forecast_evaluation.png
│   ├── har_rv_parameters.csv
│   ├── volatility_regimes.csv
│   └── forecast_evaluation.csv
│
└── README.md
```

---

## Key Results

- EGARCH outperforms GARCH(1,1) during high-volatility
  regimes by capturing the leverage effect in equity returns
- HAR-RV provides superior long-horizon forecasts (5-day,
  22-day) compared to GARCH-family models
- Rolling 63-day GARCH estimates show faster regime
  adaptation than 252-day windows during market stress
- QLIKE loss confirms EGARCH as the preferred model for
  short-horizon risk forecasting applications

---

## Applications

- Short-horizon VaR and Expected Shortfall estimation
- Options pricing and volatility surface calibration
- Portfolio risk monitoring and drawdown alerts
- Regulatory capital modeling under FRTB

---

## References

- Engle, R. (1982) — Autoregressive Conditional Heteroskedasticity
- Nelson, D. (1991) — Conditional Heteroskedasticity in Asset Returns
- Corsi, F. (2009) — A Simple Approximate Long-Memory Model (HAR-RV)
- Andersen and Bollerslev (1998) — Answering the Skeptics
