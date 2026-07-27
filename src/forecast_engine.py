"""
forecast_engine.py
==================
Walk-forward volatility forecasting engine and accuracy evaluation.
RMSE, MAE, QLIKE metrics for model comparison.

Author : Niraj Neupane | github.com/nirajneupane17
Series : Quant Trading Projects — Project 5 of 20
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from garch_models import (garch11_fit, garch11_variance, egarch_variance,
                           gjr_garch_variance, har_rv_variance, markov_switching_variance)


def walk_forward_forecast(returns: np.ndarray, window: int = 250) -> pd.DataFrame:
    """
    Walk-forward 1-day-ahead volatility forecast from all 5 models.
    Each day: fit model on trailing window → forecast tomorrow → record.
    Zero look-ahead bias. Gold standard for real forecasting evaluation.
    """
    T = len(returns); results = []
    for i in range(window, T):
        w = returns[i-window:i]; rn = returns[i]
        row = {'actual_return': rn, 'actual_vol': abs(rn) * 100}
        # GARCH(1,1)
        try:
            p = garch11_fit(w); h = garch11_variance(w, p)
            row['garch_1d'] = np.sqrt(max(p[0] + p[1]*w[-1]**2 + p[2]*h[-1], 1e-10)) * 100
        except: row['garch_1d'] = np.std(w) * 100
        # GJR-GARCH
        hg = gjr_garch_variance(w); ind = 1.0 if w[-1] < 0 else 0.0
        row['gjr_1d'] = np.sqrt(max(np.var(w)*0.02 + 0.06*w[-1]**2 + 0.08*ind*w[-1]**2 + 0.90*hg[-1], 1e-10)) * 100
        # EGARCH
        he = egarch_variance(w); sig = np.sqrt(he[-1]); z = w[-1] / sig
        lh1 = -0.10 + 0.98*np.log(he[-1]) + 0.10*(abs(z)-np.sqrt(2/np.pi)) - 0.05*z
        row['egarch_1d'] = np.sqrt(np.exp(lh1)) * 100
        # HAR-RV
        rv1 = w[-1]**2; rv5 = np.mean(w[-5:]**2); rv22 = np.mean(w[-22:]**2)
        row['har_1d'] = np.sqrt(max(0.0001 + 0.35*rv1 + 0.30*rv5 + 0.22*rv22, 1e-10)) * 100
        # MS-GARCH
        h_ms, _ = markov_switching_variance(w)
        row['ms_1d'] = np.sqrt(h_ms[-1]) * 100
        results.append(row)
    return pd.DataFrame(results)


def forecast_accuracy(forecasts: pd.DataFrame) -> pd.DataFrame:
    """RMSE, MAE, QLIKE for all model columns vs actual_vol."""
    av = forecasts['actual_vol'].values
    model_cols = [c for c in forecasts.columns if c.endswith('_1d')]
    rows = {}
    for col in model_cols:
        f = forecasts[col].values
        rows[col.replace('_1d', '')] = {
            'RMSE': round(float(np.sqrt(np.mean((f - av)**2))), 4),
            'MAE':  round(float(np.mean(np.abs(f - av))), 4),
            'QLIKE': round(float(np.mean(av**2/f**2 + np.log(f**2))), 4)
        }
    return pd.DataFrame(rows).T.sort_values('RMSE')


if __name__ == '__main__':
    df = pd.read_csv('data/garch_estimates.csv')
    r = df['return_pct'].values / 100
    print("Running walk-forward backtest (may take ~60 seconds)...")
    oos = walk_forward_forecast(r, window=250)
    acc = forecast_accuracy(oos)
    print("Forecast Accuracy:"); print(acc.to_string())
    print("forecast_engine.py OK")
