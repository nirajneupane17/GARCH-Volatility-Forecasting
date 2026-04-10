"""
rolling_vol.py
==============
Rolling volatility estimation and regime classification.
Author : Niraj Neupane | github.com/nirajneupane17
"""
import numpy as np
import pandas as pd


def rolling_volatility(returns, windows=[21,63,126,252], annualise=True):
    f = np.sqrt(252) if annualise else 1.0
    return pd.DataFrame({f'{w}d_vol': returns.rolling(w).std()*f for w in windows})


def ewma_volatility(returns, span=63, annualise=True):
    f = np.sqrt(252) if annualise else 1.0
    return returns.ewm(span=span).std() * f


def classify_regimes(vol_series, low_pct=0.33, high_pct=0.67):
    lo = vol_series.quantile(low_pct)
    hi = vol_series.quantile(high_pct)
    def _label(v):
        if v < lo:   return 'Low'
        elif v < hi: return 'Medium'
        else:        return 'High'
    return vol_series.dropna().apply(_label)


def regime_statistics(returns, vol_series):
    regimes = classify_regimes(vol_series)
    aligned = returns.loc[regimes.index]
    rows = []
    for r in ['Low','Medium','High']:
        mask = regimes == r
        ret = aligned[mask]
        rows.append({'regime': r, 'count': len(ret),
                     'mean_return': round(ret.mean()*252, 4),
                     'annual_vol':  round(ret.std()*np.sqrt(252), 4),
                     'sharpe':      round((ret.mean()/ret.std())*np.sqrt(252), 3)})
    return pd.DataFrame(rows).set_index('regime')
