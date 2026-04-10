"""
garch_models.py
===============
GARCH family volatility models — GARCH(1,1), EGARCH, GJR-GARCH.
Author : Niraj Neupane | github.com/nirajneupane17
"""
import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch(returns, p=1, q=1, dist='Normal'):
    r = returns * 100
    m = arch_model(r, vol='Garch', p=p, q=q, dist=dist, mean='Constant')
    f = m.fit(disp='off')
    a = f.params.get('alpha[1]', 0)
    b = f.params.get('beta[1]',  0)
    return {'model_type': f'GARCH({p},{q})', 'fit': f, 'params': f.params,
            'persistence': round(a+b, 6), 'aic': round(f.aic, 2),
            'bic': round(f.bic, 2), 'cond_vol': f.conditional_volatility / 100}


def fit_egarch(returns, p=1, q=1, dist='Normal'):
    r = returns * 100
    m = arch_model(r, vol='EGARCH', p=p, q=q, dist=dist, mean='Constant')
    f = m.fit(disp='off')
    g = f.params.get('gamma[1]', None)
    return {'model_type': f'EGARCH({p},{q})', 'fit': f, 'params': f.params,
            'gamma': g, 'leverage': g < 0 if g is not None else None,
            'aic': round(f.aic, 2), 'bic': round(f.bic, 2),
            'cond_vol': f.conditional_volatility / 100}


def fit_gjr_garch(returns, p=1, o=1, q=1, dist='Normal'):
    r = returns * 100
    m = arch_model(r, vol='GARCH', p=p, o=o, q=q, dist=dist, mean='Constant')
    f = m.fit(disp='off')
    return {'model_type': f'GJR-GARCH({p},{o},{q})', 'fit': f, 'params': f.params,
            'aic': round(f.aic, 2), 'bic': round(f.bic, 2),
            'cond_vol': f.conditional_volatility / 100}


def garch_forecast(fit_result, horizon=22):
    f = fit_result['fit']
    fc = f.forecast(horizon=horizon)
    vol_d = np.sqrt(fc.variance.iloc[-1]) / 100
    vol_a = vol_d * np.sqrt(252)
    return pd.DataFrame({'horizon': range(1, horizon+1),
                          'forecast_vol_daily':  vol_d.values,
                          'forecast_vol_annual': vol_a.values}).set_index('horizon')


def compare_models(returns):
    models = {'GARCH(1,1)': fit_garch(returns),
               'EGARCH':     fit_egarch(returns),
               'GJR-GARCH':  fit_gjr_garch(returns)}
    rows = [{'model': n, 'AIC': r['aic'], 'BIC': r['bic']} for n, r in models.items()]
    return pd.DataFrame(rows).set_index('model').sort_values('AIC')
