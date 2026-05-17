#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared statistical helpers for analysis scripts."""

from __future__ import annotations

import hashlib
import math
from typing import Callable, Dict, List, Tuple

import numpy as np


BOOTSTRAP_N = 400
PERMUTATION_N = 400


def rankdata(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=float)
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and v[order[j + 1]] == v[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def spearman_rho(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    rx = rankdata(xv)
    ry = rankdata(yv)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_p_value(rho: float, n: int) -> float:
    if not math.isfinite(rho) or n < 3:
        return float("nan")
    if abs(rho) >= 1.0:
        return 0.0
    t = abs(rho) * math.sqrt((n - 2) / max(1e-12, 1.0 - rho * rho))
    try:
        from scipy import stats as scipy_stats  # type: ignore

        p = 2.0 * (1.0 - float(scipy_stats.t.cdf(t, df=n - 2)))
        return min(1.0, max(0.0, p))
    except Exception:
        z = t
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
        return min(1.0, max(0.0, p))


def kendall_tau_b(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                ties_x += 1
                ties_y += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


def _stable_seed(tag: str, x: List[float], y: List[float]) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(tag.encode("utf-8"))
    h.update(b"\0")
    h.update(np.asarray(x, dtype=float).tobytes())
    h.update(b"\0")
    h.update(np.asarray(y, dtype=float).tobytes())
    return int.from_bytes(h.digest(), "little", signed=False) & 0xFFFFFFFF


def bootstrap_ci(
    stat_fn: Callable[[List[float], List[float]], float],
    x: List[float],
    y: List[float],
    n_boot: int = BOOTSTRAP_N,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    n = len(x)
    if n < 3 or n_boot < 10:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    xs = list(map(float, x))
    ys = list(map(float, y))
    values: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = [xs[int(i)] for i in idx]
        yb = [ys[int(i)] for i in idx]
        value = float(stat_fn(xb, yb))
        if math.isfinite(value):
            values.append(value)
    if len(values) < 10:
        return (float("nan"), float("nan"))
    arr = np.asarray(values, dtype=float)
    return (
        float(np.quantile(arr, alpha / 2.0)),
        float(np.quantile(arr, 1.0 - alpha / 2.0)),
    )


def kendall_p_value_permutation(
    x: List[float],
    y: List[float],
    tau_obs: float,
    n_perm: int = PERMUTATION_N,
    seed: int = 0,
) -> float:
    n = len(x)
    if n < 3 or n_perm < 10 or not math.isfinite(tau_obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    xs = [float(v) for v in x]
    ys = np.asarray(y, dtype=float)
    target = abs(float(tau_obs))
    hits = 1
    for _ in range(n_perm):
        perm = rng.permutation(n)
        tau = kendall_tau_b(xs, [float(ys[int(i)]) for i in perm])
        if math.isfinite(tau) and abs(tau) >= target:
            hits += 1
    return hits / float(n_perm + 1)


def compute_trend_stats(x: List[float], y: List[float]) -> Dict[str, float]:
    n = len(x)
    rho = spearman_rho(x, y)
    p_rho = spearman_p_value(rho, n)
    rho_ci_lo, rho_ci_hi = bootstrap_ci(
        lambda a, b: spearman_rho(a, b),
        x,
        y,
        seed=_stable_seed("rho", x, y),
    )
    tau = kendall_tau_b(x, y)
    p_tau = kendall_p_value_permutation(
        x,
        y,
        tau,
        seed=_stable_seed("tau_p", x, y),
    )
    tau_ci_lo, tau_ci_hi = bootstrap_ci(
        lambda a, b: kendall_tau_b(a, b),
        x,
        y,
        seed=_stable_seed("tau_ci", x, y),
    )
    return {
        "n": float(n),
        "rho": float(rho),
        "p_rho": float(p_rho),
        "rho_ci_lo": float(rho_ci_lo),
        "rho_ci_hi": float(rho_ci_hi),
        "tau": float(tau),
        "p_tau": float(p_tau),
        "tau_ci_lo": float(tau_ci_lo),
        "tau_ci_hi": float(tau_ci_hi),
    }


def theil_sen_slope(x: List[float], y: List[float]) -> float:
    slopes: List[float] = []
    n = len(x)
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx == 0:
                continue
            slopes.append((y[j] - y[i]) / dx)
    if not slopes:
        return float("nan")
    return float(np.median(np.asarray(slopes, dtype=float)))


def robust_slope_irls(x: List[float], y: List[float], method: str = "huber") -> float:
    if len(x) < 2:
        return float("nan")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    w = np.ones_like(x_arr, dtype=float)
    c = 1.345 if method == "huber" else 4.685
    for _ in range(20):
        w_sum = np.sum(w)
        if w_sum == 0:
            return float("nan")
        x_bar = np.sum(w * x_arr) / w_sum
        y_bar = np.sum(w * y_arr) / w_sum
        num = np.sum(w * (x_arr - x_bar) * (y_arr - y_bar))
        den = np.sum(w * (x_arr - x_bar) ** 2)
        slope = num / den if den != 0 else 0.0
        intercept = y_bar - slope * x_bar
        resid = y_arr - (slope * x_arr + intercept)
        mad = np.median(np.abs(resid - np.median(resid)))
        scale = 1.4826 * mad
        if scale <= 1e-12:
            scale = float(np.std(resid)) if np.std(resid) > 0 else 1.0
        u = resid / scale
        if method == "huber":
            w = np.where(np.abs(u) <= c, 1.0, c / np.abs(u))
        else:
            w = np.where(np.abs(u) < c, (1 - (u / c) ** 2) ** 2, 0.0)
    return float(slope)
