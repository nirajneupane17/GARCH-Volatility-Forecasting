"""
garch_models.py
===============
GARCH family volatility models — GARCH(1,1), EGARCH, GJR-GARCH,
IGARCH, HAR-RV, and Markov-Switching volatility.

Author : Niraj Neupane | github.com/nirajneupane17
Series : Quant Trading Projects — Project 5 of 20
"""
import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict


def garch11_fit(returns: np.ndarray) -> np.ndarray:
    """
    Fit GARCH(1,1) by maximum likelihood.
    Model: h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}
    Returns (omega, alpha, beta).
    """
    def neg_ll(p):
        w, a, b = p
        if w <= 0 or a <= 0 or b <= 0 or a + b >= 1: return 1e10
        n = len(returns); h = np.var(returns) * np.ones(n)
        for t in range(1, n): h[t] = w + a * returns[t-1]**2 + b * h[t-1]
        h = np.maximum(h, 1e-10)
        return 0.5 * np.sum(np.log(h) + returns**2 / h)
    res = minimize(neg_ll, [np.var(returns) * 0.05, 0.09, 0.90],
                   method='L-BFGS-B',
                   bounds=[(1e-8, 1), (1e-6, 0.5), (1e-6, 0.999)])
    return res.x


def garch11_variance(returns: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Compute GARCH(1,1) conditional variance series."""
    w, a, b = params; n = len(returns)
    h = np.zeros(n); h[0] = np.var(returns)
    for t in range(1, n): h[t] = w + a * returns[t-1]**2 + b * h[t-1]
    return np.maximum(h, 1e-10)


def garch11_forecast(returns: np.ndarray, params: np.ndarray,
                      h_ahead: int = 10) -> np.ndarray:
    """
    Multi-step ahead GARCH(1,1) variance forecast.
    h_{t+k} = omega/(1-alpha-beta) + (alpha+beta)^k * (h_t - omega/(1-alpha-beta))
    Returns converges to long-run variance as k increases.
    """
    w, a, b = params; h = garch11_variance(returns, params)
    lterm = w / (1 - a - b)  # long-run variance
    h_t = h[-1]
    forecasts = np.array([lterm + (h_t - lterm) * (a + b)**k for k in range(1, h_ahead + 1)])
    return np.maximum(forecasts, 1e-10)


def egarch_variance(returns: np.ndarray,
                     omega: float = -0.10, alpha: float = 0.10,
                     gamma: float = -0.05, beta: float = 0.98) -> np.ndarray:
    """
    EGARCH (Nelson 1991) conditional variance.
    log(h_t) = omega + beta*log(h_{t-1}) + alpha*(|z_{t-1}| - E|z|) + gamma*z_{t-1}
    gamma < 0 → negative shocks increase vol more (leverage effect).
    """
    n = len(returns); lh = np.zeros(n); lh[0] = np.log(np.var(returns))
    for t in range(1, n):
        sig = np.sqrt(np.exp(lh[t-1])); z = returns[t-1] / sig
        lh[t] = omega + beta * lh[t-1] + alpha * (np.abs(z) - np.sqrt(2/np.pi)) + gamma * z
    return np.maximum(np.exp(lh), 1e-10)


def gjr_garch_variance(returns: np.ndarray,
                        omega: float = None, alpha: float = 0.06,
                        beta: float = 0.90, gamma: float = 0.08) -> np.ndarray:
    """
    GJR-GARCH (Glosten-Jagannathan-Runkle 1993).
    h_t = omega + alpha*r_{t-1}^2 + gamma*I(r_{t-1}<0)*r_{t-1}^2 + beta*h_{t-1}
    gamma > 0 → negative returns raise vol more than positive (leverage effect).
    Effective alpha for negative shocks: alpha + gamma.
    """
    if omega is None: omega = np.var(returns) * 0.02
    n = len(returns); h = np.zeros(n); h[0] = np.var(returns)
    for t in range(1, n):
        ind = 1.0 if returns[t-1] < 0 else 0.0
        h[t] = omega + alpha * returns[t-1]**2 + gamma * ind * returns[t-1]**2 + beta * h[t-1]
    return np.maximum(h, 1e-10)


def har_rv_variance(returns: np.ndarray,
                     c: float = 0.0001, beta1: float = 0.35,
                     beta5: float = 0.30, beta22: float = 0.22) -> np.ndarray:
    """
    HAR-RV (Corsi 2009) — Heterogeneous AutoRegression of Realized Volatility.
    RV_t = c + beta1*RV_{t-1} + beta5*RV_{t-5:t} + beta22*RV_{t-22:t} + epsilon
    Uses squared returns as proxy for daily realized variance.
    Outperforms GARCH at horizons > 5 days due to multi-scale structure.
    """
    rv = returns**2; n = len(rv); h = np.zeros(n); h[:22] = np.var(returns)
    for t in range(22, n):
        h[t] = c + beta1 * rv[t-1] + beta5 * rv[t-5:t].mean() + beta22 * rv[t-22:t].mean()
    return np.maximum(h, 1e-10)


def markov_switching_variance(returns: np.ndarray,
                               window: int = 30,
                               threshold: float = 0.015) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple 2-state Markov-Switching variance model.
    State 0: calm (low volatility regime)
    State 1: turbulent (high volatility regime)
    State assigned based on rolling standard deviation vs threshold.
    Returns (variance_series, state_series).
    """
    n = len(returns); states = np.zeros(n, dtype=int)
    for t in range(window, n):
        if np.std(returns[max(0, t-window):t]) > threshold:
            states[t] = 1
    sigma_calm  = np.std(returns[states == 0]) if np.any(states == 0) else 0.008
    sigma_turb  = np.std(returns[states == 1]) if np.any(states == 1) else 0.018
    h = np.where(states == 0, sigma_calm**2, sigma_turb**2)
    return np.maximum(h, 1e-10), states


def news_impact_curve(params: np.ndarray, h0: float,
                       model: str = 'garch',
                       shocks: np.ndarray = None) -> np.ndarray:
    """
    Compute news impact curve — h(z) as a function of shock z.
    Shows how volatility responds asymmetrically to positive vs negative shocks.
    """
    if shocks is None: shocks = np.linspace(-0.05, 0.05, 300)
    w, a, b = params[:3]
    if model == 'garch':
        return np.array([w + a * z**2 + b * h0 for z in shocks])
    elif model == 'gjr':
        gamma = params[3] if len(params) > 3 else 0.08
        return np.array([w + a*z**2 + gamma*(1.0 if z < 0 else 0)*z**2 + b*h0 for z in shocks])
    return np.array([w + a * z**2 + b * h0 for z in shocks])


def garch_persistence(params: np.ndarray) -> Dict:
    """
    Compute GARCH(1,1) persistence metrics.
    Persistence = alpha + beta. Half-life = log(0.5) / log(persistence).
    Unconditional variance = omega / (1 - alpha - beta).
    """
    w, a, b = params
    persist = a + b
    half_life = np.log(0.5) / np.log(persist) if persist < 1 else np.inf
    uncond_var = w / (1 - persist) if persist < 1 else np.inf
    return {'alpha': round(a, 6), 'beta': round(b, 6),
            'persistence': round(persist, 6),
            'half_life_days': round(half_life, 1),
            'unconditional_vol_pct': round(np.sqrt(uncond_var) * 100, 4)}


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('data/garch_estimates.csv')
    r = df['return_pct'].values / 100
    params = garch11_fit(r)
    metrics = garch_persistence(params)
    print("GARCH(1,1) Parameters:"); [print(f"  {k}: {v}") for k, v in metrics.items()]
    print("garch_models.py OK")
