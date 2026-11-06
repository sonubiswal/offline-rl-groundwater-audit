"""Interval utilities. Pure numpy/scipy; no project dependencies."""
import numpy as np
from scipy import stats


def t_interval(x, conf=0.95):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()) if n else np.nan, np.nan, np.nan
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * se
    return float(m), float(m - h), float(m + h)


def _pct(boot, conf):
    a = (1 - conf) / 2
    return np.quantile(boot, [a, 1 - a])


def episode_bootstrap(diff_eps, n_boot=5000, conf=0.95, rng=None):
    rng = rng or np.random.default_rng(2026)
    d = np.asarray(diff_eps, float)
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    boot = rng.choice(d, (n_boot, len(d)), replace=True).mean(1)
    lo, hi = _pct(boot, conf)
    return float(d.mean()), float(lo), float(hi)


def seed_bootstrap(seed_means, n_boot=5000, conf=0.95, rng=None):
    rng = rng or np.random.default_rng(2026)
    s = np.asarray(seed_means, float)
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    boot = rng.choice(s, (n_boot, len(s)), replace=True).mean(1)
    lo, hi = _pct(boot, conf)
    return float(s.mean()), float(lo), float(hi)


def hierarchical_bootstrap(diff_by_seed, n_boot=5000, conf=0.95, rng=None):
    rng = rng or np.random.default_rng(2026)
    k = len(diff_by_seed)
    if k == 0:
        return np.nan, np.nan, np.nan
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, k)
        boot[b] = np.mean([rng.choice(diff_by_seed[i],
                                      len(diff_by_seed[i])).mean() for i in idx])
    lo, hi = _pct(boot, conf)
    m = float(np.mean([d.mean() for d in diff_by_seed]))
    return m, float(lo), float(hi)