"""Shared utilities for plotting optimization vs. reference results."""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


# Optimization columns
OPT_LAMBDA_COL = "lambda"
OPT_X_COL = "rel_connected_time"
OPT_Y_COL = "mean_capacity"

# Reference columns
REF_X_COL = "rel_connected_time"
REF_Y_COL = "mean_capacity"

# Reference parameter-set identifier columns
REF_GROUP_COLS = ["a3_ttt_ms", "a3_hys_db", "a3_off_dbm"]


@dataclass(frozen=True)
class Curve:
    """One curve of mean values."""

    kind: str  # "opt" or "topq"
    pct: float | None  # None for opt; percentile q for reference
    n_sel: int  # sample count used to form the mean
    lam: np.ndarray
    x_mean: np.ndarray
    y_mean: np.ndarray


def get_default_optim_result_path(dataset_dir: str) -> str:
    """Return default optimization metrics Parquet path."""
    return os.path.join(
        os.getcwd(),
        dataset_dir,
        "optim_results",
        "gp_optim_result_metrics.parquet",
    )


def get_default_reference_result_path(dataset_dir: str) -> str:
    """Return default reference metrics Parquet path."""
    return os.path.join(
        os.getcwd(),
        dataset_dir,
        "reference_results",
        "reference_result_metrics.parquet",
    )


def get_default_plot_path() -> str:
    """Return default output directory for plots."""
    plots_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert selected columns to numeric in-place."""
    df = df.copy()
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    return df


def _pareto_frontier_max_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Boolean mask selecting achieved Pareto frontier for maximizing (x, y).
    """
    n = x.size
    if n == 0:
        return np.zeros((0,), dtype=bool)

    order = np.argsort(x, kind="mergesort")
    y_sorted = y[order]

    keep_sorted = np.zeros(n, dtype=bool)
    best_y = -np.inf
    for idx in range(n - 1, -1, -1):
        if y_sorted[idx] > best_y:
            keep_sorted[idx] = True
            best_y = y_sorted[idx]

    keep = np.zeros(n, dtype=bool)
    keep[order] = keep_sorted
    return keep


def load_optimization_metrics(path: str) -> pd.DataFrame:
    """Load optimization metrics dataset."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Optimization dataset not found: {path}")

    df = pd.read_parquet(path)
    required = [OPT_LAMBDA_COL, OPT_X_COL, OPT_Y_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required optimization columns: {missing}")

    df = _to_numeric(df, required)
    df = df.dropna(subset=required)
    return df


def load_reference_parameter_aggregation(path: str) -> pd.DataFrame:
    """
    Load reference metrics dataset and aggregate per parameter set.

    The resulting rows correspond to aggregated parameter-set means:
        k = (a3_ttt_ms, a3_hys_db, a3_off_dbm)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Reference dataset not found: {path}")

    df = pd.read_parquet(path)
    required = REF_GROUP_COLS + [REF_X_COL, REF_Y_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required reference columns: {missing}")

    df = _to_numeric(df, required)
    df = df.dropna(subset=required)

    return df.groupby(REF_GROUP_COLS, as_index=False)[[REF_X_COL, REF_Y_COL]].mean()


def load_optimization_curve(path: str) -> Curve:
    """
    Build optimization curve by grouping all rows by lambda.

    Only mean values are returned.
    """
    df = load_optimization_metrics(path)

    lambdas: list[float] = []
    x_mean: list[float] = []
    y_mean: list[float] = []
    ns: list[int] = []

    grouped = df.groupby(OPT_LAMBDA_COL, as_index=False)
    for lam_value, grp in grouped:
        xs = grp[OPT_X_COL].to_numpy(dtype=float)
        ys = grp[OPT_Y_COL].to_numpy(dtype=float)

        xs = xs[np.isfinite(xs)]
        ys = ys[np.isfinite(ys)]
        if xs.size == 0 or ys.size == 0:
            continue

        lambdas.append(float(lam_value))  # type: ignore[arg-type]
        x_mean.append(float(xs.mean()))
        y_mean.append(float(ys.mean()))
        ns.append(int(min(xs.size, ys.size)))

    order = np.argsort(np.asarray(lambdas, dtype=float))
    lam = np.asarray(lambdas, dtype=float)[order]

    return Curve(
        kind="opt",
        pct=None,
        n_sel=int(np.max(np.asarray(ns, dtype=int))) if ns else 0,
        lam=lam,
        x_mean=np.asarray(x_mean, dtype=float)[order],
        y_mean=np.asarray(y_mean, dtype=float)[order],
    )


def reference_curve_for_quantile_top_tail(
    df_agg: pd.DataFrame,
    lambdas: np.ndarray,
    pct: float,
) -> Curve:
    """
    For each lambda and percentile q, compute
        s_k(lambda) = L_r,k - lambda * (1 - L_c,k)

    select the upper-tail set
        S_q(lambda) = {k | s_k(lambda) >= Q_s(q, lambda)}

    and return the average over the selected parameter sets:
        (bar L_c,q(lambda), bar L_r,q(lambda)).
    """
    l_r_k = df_agg[REF_Y_COL].to_numpy(dtype=float)
    l_c_k = df_agg[REF_X_COL].to_numpy(dtype=float)

    if l_r_k.size == 0:
        raise RuntimeError("No reference parameter sets available after aggregation.")

    q = float(np.clip(float(pct), 0.0, 100.0))

    x_mean = np.empty(lambdas.size, dtype=float)
    y_mean = np.empty(lambdas.size, dtype=float)
    n_sel_vec = np.empty(lambdas.size, dtype=int)

    for idx, lam in enumerate(lambdas):
        s_k = l_r_k - float(lam) * (1.0 - l_c_k)
        q_s = float(np.percentile(s_k, q))
        selected = np.flatnonzero(s_k >= q_s)

        if selected.size == 0:
            selected = np.array([int(np.argmax(s_k))], dtype=int)

        xs = l_c_k[selected]
        ys = l_r_k[selected]

        x_mean[idx] = float(xs.mean())
        y_mean[idx] = float(ys.mean())
        n_sel_vec[idx] = int(selected.size)

    return Curve(
        kind="topq",
        pct=q,
        n_sel=int(np.min(n_sel_vec)) if n_sel_vec.size else 0,
        lam=lambdas.copy(),
        x_mean=x_mean,
        y_mean=y_mean,
    )


def reference_pareto_bound(df_agg: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return achieved reference Pareto bound from aggregated parameter sets."""
    conn = df_agg[REF_X_COL].to_numpy(dtype=float)
    cap = df_agg[REF_Y_COL].to_numpy(dtype=float)

    keep = _pareto_frontier_max_xy(conn, cap)
    conn_front = conn[keep]
    cap_front = cap[keep]

    order = np.argsort(conn_front)
    return conn_front[order], cap_front[order]
