"""Plot-data builders and publication-quality figures.

Plot data is always written separately from the image.  This keeps every
population, denominator and lambda=0 result auditable even where the figure
intentionally omits counts or cannot display zero on a logarithmic axis.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

POSITIVE_LAMBDA_MIN = 0.1
POSITIVE_LAMBDA_MAX = 1000.0
OUTAGE_COMPONENTS = (
    "successful_ho_exec_seconds",
    "hof_related_seconds",
    "rlf_no_ho_seconds",
)
LINE_STYLES = {
    "MILP (genie oracle)": "-",
    "A3 global (ID-Val)": "--",
    "A3 speed-aware (ID-Val)": "-.",
    "A3 split oracle (non-deployable)": ":",
    "No-HO": (0, (5, 2)),
}
CDF_CURVE_STYLES = {
    ("prep_end", "qin"): "-",
    ("prep_end", "qout"): "--",
    ("exec_end", "qin"): "-.",
    ("exec_end", "qout"): ":",
}
CDF_CURVE_LABELS = {
    ("prep_end", "qin"): "HO prep end · Qin (80 ms)",
    ("prep_end", "qout"): "HO prep end · Qout (200 ms)",
    ("exec_end", "qin"): "HO exec end · Qin (80 ms)",
    ("exec_end", "qout"): "HO exec end · Qout (200 ms)",
}


def _series_name(frame: pd.DataFrame) -> pd.Series:
    level = frame.get("selection_level", pd.Series(None, index=frame.index))
    labels = pd.Series(index=frame.index, dtype="object")
    labels.loc[frame["method"] == "milp"] = "MILP (genie oracle)"
    labels.loc[frame["method"] == "no_ho"] = "No-HO"
    a3_labels = level.map(
        {
            "global": "A3 global (ID-Val)",
            "speed_aware": "A3 speed-aware (ID-Val)",
            "split_oracle": "A3 split oracle (non-deployable)",
        }
    )
    labels.loc[frame["method"] == "a3"] = a3_labels
    if labels.isna().any():
        unknown = frame.loc[labels.isna(), ["method", "selection_level"]]
        raise ValueError(
            f"unknown reporting series: {unknown.drop_duplicates().to_dict('records')}"
        )
    return labels


def build_event_rate_plot_data(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate HO/HOF/RLF counts while retaining lambda=0 rows."""
    required = {
        "split",
        "method",
        "lambda",
        "simulated_time_s",
        "num_ho",
        "num_hof",
        "num_rlf",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"event-rate metrics missing columns: {missing}")
    frame = metrics.copy()
    if "selection_level" not in frame:
        frame["selection_level"] = None
    frame["series"] = _series_name(frame)
    group = ["split", "method", "selection_level", "series", "lambda"]
    sums = frame.groupby(group, dropna=False, as_index=False)[
        ["simulated_time_s", "num_ho", "num_hof", "num_rlf"]
    ].sum()
    for event in ("ho", "hof", "rlf"):
        sums[f"{event}_per_s"] = sums[f"num_{event}"] / sums["simulated_time_s"]
    return sums.sort_values(["split", "series", "lambda"]).reset_index(drop=True)


def build_operating_point_plot_data(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rate/outage/objective for every explicit baseline level."""
    required = {
        "split",
        "method",
        "selection_level",
        "lambda",
        "mean_capacity",
        "rel_connected_time",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"operating-point metrics missing columns: {missing}")
    frame = metrics.copy()
    frame["series"] = _series_name(frame)
    group = ["split", "method", "selection_level", "series", "lambda"]
    result = frame.groupby(group, dropna=False, as_index=False)[
        ["mean_capacity", "rel_connected_time"]
    ].mean()
    result["outage"] = 1.0 - result["rel_connected_time"]
    result["objective"] = result["mean_capacity"] - result["lambda"] * result["outage"]
    return result.sort_values(["split", "series", "lambda"]).reset_index(drop=True)


def build_handover_sinr_cdf_data(attempts: pd.DataFrame) -> pd.DataFrame:
    """Build boundary CDFs from the exact source/target L1 RLM values."""
    populations = (
        ("prep_end", "source", None),
        ("exec_end", "target", {"success", "hof_exec"}),
    )
    rows: list[pd.DataFrame] = []
    work = attempts.copy()
    if "selection_level" not in work:
        work["selection_level"] = None
    work["series"] = _series_name(work)
    for phase, cell_role, outcomes in populations:
        step_column = f"{phase}_step"
        for window in ("qin", "qout"):
            value_column = f"{phase}_l1_{window}_sinr_db"
            required = {step_column, value_column, "outcome"}
            missing = sorted(required - set(work.columns))
            if missing:
                raise ValueError(f"attempt table missing CDF columns: {missing}")
            selected = work[
                work[step_column].notna() & work[value_column].notna()
            ].copy()
            if outcomes is not None:
                selected = selected[selected["outcome"].isin(outcomes)]
            group_columns = [
                "split",
                "method",
                "selection_level",
                "series",
                "lambda",
            ]
            for identity, group in selected.groupby(group_columns, dropna=False):
                values = np.sort(group[value_column].to_numpy(dtype=float))
                if not len(values):
                    continue
                chunk = pd.DataFrame(
                    {
                        **{name: value for name, value in zip(group_columns, identity)},
                        "phase": phase,
                        "cell_role": cell_role,
                        "rlm_window": window,
                        "sinr_db": values,
                        "cdf": np.arange(1, len(values) + 1) / len(values),
                        "population_count": len(values),
                        "success_count": int((group["outcome"] == "success").sum()),
                        "failure_count": int((group["outcome"] != "success").sum()),
                        "hof_prep_count": int((group["outcome"] == "hof_prep").sum()),
                        "hof_cmd_count": int((group["outcome"] == "hof_cmd").sum()),
                        "hof_exec_count": int((group["outcome"] == "hof_exec").sum()),
                        "censored_count": int((group["outcome"] == "censored").sum()),
                        "rlm_not_evaluated_count": int(
                            (
                                ~group[f"{phase}_rlm_evaluated"]
                                .fillna(False)
                                .astype(bool)
                            ).sum()
                        ),
                    }
                )
                rows.append(chunk)
    columns = [
        "split",
        "method",
        "selection_level",
        "series",
        "lambda",
        "phase",
        "cell_role",
        "rlm_window",
        "sinr_db",
        "cdf",
        "population_count",
        "success_count",
        "failure_count",
        "hof_prep_count",
        "hof_cmd_count",
        "hof_exec_count",
        "censored_count",
        "rlm_not_evaluated_count",
    ]
    return (
        pd.concat(rows, ignore_index=True)[columns]
        if rows
        else pd.DataFrame(columns=columns)
    )


def build_outage_stacked_plot_data(outage: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the three disjoint, named outage components in UE-seconds."""
    required = {"split", "method", "lambda", *OUTAGE_COMPONENTS, "outage_seconds"}
    missing = sorted(required - set(outage.columns))
    if missing:
        raise ValueError(f"outage table missing columns: {missing}")
    frame = outage.copy()
    if "selection_level" not in frame:
        frame["selection_level"] = None
    frame["series"] = _series_name(frame)
    group = ["split", "method", "selection_level", "series", "lambda"]
    result = frame.groupby(group, dropna=False, as_index=False)[
        [*OUTAGE_COMPONENTS, "outage_seconds"]
    ].sum()
    components = result[list(OUTAGE_COMPONENTS)].sum(axis=1)
    if not np.allclose(components, result["outage_seconds"], rtol=0.0, atol=1e-12):
        bad = result.loc[~np.isclose(components, result["outage_seconds"], atol=1e-12)]
        raise ValueError(
            f"outage components do not conserve total: {bad.head().to_dict('records')}"
        )
    return result.sort_values(["split", "lambda", "series"]).reset_index(drop=True)


def _save_figure(fig, stem: Path, formats: Sequence[str]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=200, bbox_inches="tight")


def plot_event_rates(
    data: pd.DataFrame, output_stem: Path, *, formats: Sequence[str] = ("png", "pdf")
) -> None:
    positive = data[
        (data["lambda"] >= POSITIVE_LAMBDA_MIN)
        & (data["lambda"] <= POSITIVE_LAMBDA_MAX)
    ]
    ticks = sorted(positive["lambda"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    for axis, event in zip(axes, ("ho", "hof", "rlf")):
        for series, group in positive.groupby("series"):
            ordered = group.sort_values("lambda")
            axis.plot(
                ordered["lambda"],
                ordered[f"{event}_per_s"],
                marker="o",
                label=series,
                linestyle=LINE_STYLES.get(str(series), "-"),
            )
        axis.set_xscale("log")
        axis.set_xlim(POSITIVE_LAMBDA_MIN, POSITIVE_LAMBDA_MAX)
        axis.xaxis.set_major_locator(FixedLocator(ticks))
        axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: f"{value:g}")
        )
        axis.set_xlabel(r"$\lambda$")
        axis.set_ylabel(f"{event.upper()} events / s")
        axis.grid(True, which="both", alpha=0.25)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=max(1, len(labels)),
    )
    _save_figure(fig, Path(output_stem), formats)
    plt.close(fig)


def plot_operating_points(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    formats: Sequence[str] = ("png", "pdf"),
) -> None:
    """Render rate-outage and objective comparisons with all A3 levels."""
    output_dir = Path(output_dir)
    def _plot_frontier(frame: pd.DataFrame, stem: str) -> None:
        frontier, axis = plt.subplots(figsize=(5.6, 4.0), constrained_layout=True)
        for series, group in frame.groupby("series"):
            ordered = group.sort_values("lambda")
            axis.plot(
                ordered["outage"],
                ordered["mean_capacity"],
                marker="o",
                label=series,
                linestyle=LINE_STYLES.get(str(series), "-"),
            )
        axis.set_xlabel("Outage fraction")
        axis.set_ylabel("Mean capacity [bit/s/Hz]")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
        axis.margins(x=0.08, y=0.08)
        _save_figure(frontier, output_dir / stem, formats)
        plt.close(frontier)

    _plot_frontier(data, "rate_outage_frontier")
    zoom = data[data["series"] != "No-HO"]
    if not zoom.empty:
        _plot_frontier(zoom, "rate_outage_frontier_zoom")

    positive = data[
        (data["lambda"] >= POSITIVE_LAMBDA_MIN)
        & (data["lambda"] <= POSITIVE_LAMBDA_MAX)
    ]
    objective, axis = plt.subplots(figsize=(5.6, 4.0), constrained_layout=True)
    for series, group in positive.groupby("series"):
        ordered = group.sort_values("lambda")
        axis.plot(
            ordered["lambda"],
            ordered["objective"],
            marker="o",
            label=series,
            linestyle=LINE_STYLES.get(str(series), "-"),
        )
    ticks = sorted(positive["lambda"].unique())
    axis.set_xscale("log")
    axis.set_xlim(POSITIVE_LAMBDA_MIN, POSITIVE_LAMBDA_MAX)
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
    axis.set_xlabel(r"$\lambda$")
    axis.set_ylabel("Objective")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    _save_figure(objective, output_dir / "objective_vs_lambda", formats)
    plt.close(objective)


def plot_handover_sinr_cdfs(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    q_in_db: float,
    q_out_db: float,
    formats: Sequence[str] = ("png", "pdf"),
) -> None:
    """Write one eight-lambda CDF matrix containing every available method."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for obsolete in output_dir.glob("ho_*_sinr_cdf_lambda*"):
        if obsolete.is_file():
            obsolete.unlink()

    lambda_values = sorted(data["lambda"].dropna().unique())
    if len(lambda_values) > 8:
        raise ValueError(f"CDF matrix supports at most eight lambdas, got {lambda_values}")
    series_names = list(dict.fromkeys(data["series"].astype(str)))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    series_colors = {
        name: color_cycle[index % len(color_cycle)]
        for index, name in enumerate(series_names)
    }
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.7), sharex=True, sharey=True)
    for axis, lambda_value in zip(axes.flat, lambda_values):
        population = data[data["lambda"] == lambda_value]
        for (series, phase, window), group in population.groupby(
            ["series", "phase", "rlm_window"], dropna=False
        ):
            ordered = group.sort_values("sinr_db")
            axis.plot(
                ordered["sinr_db"],
                ordered["cdf"],
                color=series_colors[str(series)],
                linestyle=CDF_CURVE_STYLES[(str(phase), str(window))],
                linewidth=1.35,
            )
        axis.axvline(q_in_db, color="#555555", linestyle=(0, (5, 2)), linewidth=1.0)
        axis.axvline(q_out_db, color="#888888", linestyle=(0, (2, 2)), linewidth=1.0)
        axis.set_title(rf"$\lambda={lambda_value:g}$")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.25)
    for axis in axes.flat[len(lambda_values) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        axis.set_xlabel("L1 SINR [dB]")
    for axis in axes[:, 0]:
        axis.set_ylabel("CDF")

    legend_handles = [
        Line2D([0], [0], color=series_colors[name], linewidth=2.0, label=name)
        for name in series_names
    ]
    legend_handles.extend(
        Line2D([0], [0], color="black", linestyle=style, label=CDF_CURVE_LABELS[key])
        for key, style in CDF_CURVE_STYLES.items()
    )
    legend_handles.extend(
        (
            Line2D(
                [0],
                [0],
                color="#555555",
                linestyle=(0, (5, 2)),
                label=f"Qin = {q_in_db:g} dB",
            ),
            Line2D(
                [0],
                [0],
                color="#888888",
                linestyle=(0, (2, 2)),
                label=f"Qout = {q_out_db:g} dB",
            ),
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.07, right=0.985, hspace=0.26)
    _save_figure(fig, output_dir / "ho_execution_sinr_cdf", formats)
    plt.close(fig)


def plot_outage_stacked(
    data: pd.DataFrame, output_stem: Path, *, formats: Sequence[str] = ("png", "pdf")
) -> None:
    lambdas = sorted(data["lambda"].unique())
    series = list(dict.fromkeys(data["series"].astype(str)))
    width = 0.8 / max(1, len(series))
    x = np.arange(len(lambdas), dtype=float)
    colors = ("#4c78a8", "#e45756", "#f2cf5b")
    labels = ("HO execution", "HOF interruption", "RLF without HOF")
    hatches = ("", "//", "xx", "..", "\\\\")
    fig, axis = plt.subplots(figsize=(12.5, 5.4))
    for series_index, series_name in enumerate(series):
        selected = data[data["series"] == series_name].set_index("lambda")
        positions = x - 0.4 + width / 2 + series_index * width
        bottom = np.zeros(len(lambdas))
        for component, color, label in zip(OUTAGE_COMPONENTS, colors, labels):
            values = np.asarray(
                [selected[component].get(value, 0.0) for value in lambdas]
            )
            axis.bar(
                positions,
                values,
                width,
                bottom=bottom,
                color=color,
                edgecolor="black",
                linewidth=0.35,
                hatch=hatches[series_index % len(hatches)],
                label=label if series_index == 0 else None,
            )
            bottom += values
    axis.set_xticks(x, [f"{value:g}" for value in lambdas])
    axis.set_xlabel(r"$\lambda$")
    axis.set_ylabel("Cumulative outage [UE-s]")
    component_handles = [
        Patch(facecolor=color, edgecolor="black", label=label)
        for color, label in zip(colors, labels)
    ]
    method_handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=hatches[index % len(hatches)],
            label=name,
        )
        for index, name in enumerate(series)
    ]
    fig.legend(
        handles=[*component_handles, *method_handles],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        fontsize=8.5,
    )
    split_names = data["split"].dropna().astype(str).unique()
    title = "Outage decomposition"
    if len(split_names) == 1:
        title += f" — {split_names[0].replace('_', ' ')}"
    fig.suptitle(title, y=0.985)
    axis.grid(True, axis="y", alpha=0.25)
    fig.subplots_adjust(top=0.75, bottom=0.13, left=0.08, right=0.99)
    _save_figure(fig, Path(output_stem), formats)
    plt.close(fig)


def publish_reports(
    *,
    metrics: pd.DataFrame,
    attempts: pd.DataFrame,
    outage: pd.DataFrame,
    output_root: Path,
    q_in_db: float,
    q_out_db: float,
    report_splits: Sequence[str] = (
        "id_train",
        "id_val",
        "id_test",
        "ood_speed",
        "ood_dynamics",
    ),
) -> None:
    """Write plot tables and figures for each report split."""
    root = Path(output_root)
    for split in report_splits:
        directory = root / "plots" / split
        directory.mkdir(parents=True, exist_ok=True)
        event_data = build_event_rate_plot_data(metrics[metrics["split"] == split])
        operating_data = build_operating_point_plot_data(
            metrics[metrics["split"] == split]
        )
        cdf_data = build_handover_sinr_cdf_data(attempts[attempts["split"] == split])
        outage_data = build_outage_stacked_plot_data(outage[outage["split"] == split])
        event_data.to_parquet(
            directory / "event_rates_vs_lambda_data.parquet", index=False
        )
        operating_data.to_parquet(
            directory / "operating_points_data.parquet", index=False
        )
        cdf_data.to_parquet(
            directory / "ho_execution_sinr_cdf_data.parquet", index=False
        )
        outage_data.to_parquet(
            directory / "outage_decomposition_vs_lambda_data.parquet", index=False
        )
        plot_event_rates(event_data, directory / "event_rates_vs_lambda")
        plot_operating_points(operating_data, directory)
        plot_handover_sinr_cdfs(cdf_data, directory, q_in_db=q_in_db, q_out_db=q_out_db)
        plot_outage_stacked(
            outage_data, directory / "outage_decomposition_vs_lambda"
        )


__all__ = [
    "OUTAGE_COMPONENTS",
    "build_event_rate_plot_data",
    "build_handover_sinr_cdf_data",
    "build_operating_point_plot_data",
    "build_outage_stacked_plot_data",
    "plot_event_rates",
    "plot_handover_sinr_cdfs",
    "plot_operating_points",
    "plot_outage_stacked",
    "publish_reports",
]
