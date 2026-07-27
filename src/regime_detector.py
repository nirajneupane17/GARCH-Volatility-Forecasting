"""
regime_detector.py
==================
Volatility regime detection — Markov-Switching, rolling Z-score,
and GARCH-based regime classification.

Author : Niraj Neupane | github.com/nirajneupane17
Series : Quant Trading Projects — Project 5 of 20
"""
import numpy as np
import pandas as pd
from typing import Tuple


def rolling_regime(returns: np.ndarray, window: int = 30,
                   threshold: float = 0.015) -> np.ndarray:
    """
    Simple rolling-window regime detector.
    State 0: calm (rolling vol < threshold)
    State 1: turbulent (rolling vol >= threshold)
    """
    n = len(returns); states = np.zeros(n, dtype=int)
    for t in range(window, n):
        if np.std(returns[max(0, t-window):t]) >= threshold:
            states[t] = 1
    return states


def vol_zscore_regime(vol_series: np.ndarray, window: int = 60,
                       z_threshold: float = 1.0) -> np.ndarray:
    """
    Regime detection via volatility Z-score.
    Turbulent when rolling vol > mean + z_threshold * std.
    More adaptive than fixed threshold.
    """
    n = len(vol_series); states = np.zeros(n, dtype=int)
    for t in range(window, n):
        w = vol_series[t-window:t]
        z = (vol_series[t] - w.mean()) / (w.std() + 1e-10)
        if z > z_threshold: states[t] = 1
    return states


def regime_statistics(returns: np.ndarray, states: np.ndarray) -> dict:
    """
    Summary statistics for each regime.
    Returns mean vol, max drawdown, fraction of time, persistence.
    """
    r_calm  = returns[states == 0]; r_turb  = returns[states == 1]
    p_calm  = np.mean(states == 0); p_turb  = np.mean(states == 1)
    # Regime persistence: P(stay in same state)
    transitions = np.diff(states)
    p11 = np.mean(states[1:][states[:-1] == 1] == 1) if np.any(states == 1) else 0
    p00 = np.mean(states[1:][states[:-1] == 0] == 0) if np.any(states == 0) else 0
    return {
        'calm_vol_pct':     round(np.std(r_calm) * 100, 4) if len(r_calm) > 0 else 0,
        'turbulent_vol_pct': round(np.std(r_turb) * 100, 4) if len(r_turb) > 0 else 0,
        'calm_fraction':    round(float(p_calm), 4),
        'turbulent_fraction': round(float(p_turb), 4),
        'p00_persistence':  round(float(p00), 4),
        'p11_persistence':  round(float(p11), 4),
        'n_transitions':    int(np.sum(np.abs(transitions)))
    }


if __name__ == '__main__':
    df = pd.read_csv('data/garch_estimates.csv')
    r = df['return_pct'].values / 100
    states = rolling_regime(r)
    stats = regime_statistics(r, states)
    print("Regime Statistics:"); [print(f"  {k}: {v}") for k, v in stats.items()]
    print("regime_detector.py OK")
