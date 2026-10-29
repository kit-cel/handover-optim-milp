from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ho_optim_milp.dataset_generator.reporting import (
    build_event_rate_plot_data,
    build_handover_sinr_cdf_data,
    build_operating_point_plot_data,
    build_outage_stacked_plot_data,
    plot_event_rates,
    plot_handover_sinr_cdfs,
    plot_operating_points,
    plot_outage_stacked,
)


def test_cdf_uses_current_phase_boundary_and_individual_rlm_windows() -> None:
    attempts = pd.DataFrame(
        {
            "split": ["id_test"] * 3,
            "method": ["milp", "a3", "a3"],
            "selection_level": ["none", "global", "global"],
            "lambda": [1.0] * 3,
            "outcome": ["success", "success", "hof_exec"],
            "prep_end_step": [4, 5, 6],
            "exec_end_step": [8, 9, 10],
            "prep_end_l1_qin_sinr_db": [-4.0, -3.0, -2.0],
            "prep_end_l1_qout_sinr_db": [-8.0, -7.0, -6.0],
            "exec_end_l1_qin_sinr_db": [-5.0, -4.0, -3.0],
            "exec_end_l1_qout_sinr_db": [-9.0, -8.0, -7.0],
            "prep_end_rlm_evaluated": [True, False, True],
            "exec_end_rlm_evaluated": [True, True, True],
        }
    )
    result = build_handover_sinr_cdf_data(attempts)

    a3_exec_qout = result[
        (result["series"] == "A3 global (ID-Val)")
        & (result["phase"] == "exec_end")
        & (result["rlm_window"] == "qout")
    ]
    assert a3_exec_qout["sinr_db"].tolist() == [-8.0, -7.0]
    assert a3_exec_qout["population_count"].unique().tolist() == [2]
    assert a3_exec_qout["success_count"].unique().tolist() == [1]
    assert a3_exec_qout["failure_count"].unique().tolist() == [1]


def test_event_plot_data_retains_lambda_zero_for_tables() -> None:
    metrics = pd.DataFrame(
        {
            "split": ["id_test", "id_test"],
            "method": ["milp", "milp"],
            "selection_level": ["none", "none"],
            "lambda": [0.0, 0.1],
            "simulated_time_s": [10.0, 10.0],
            "num_ho": [2, 1],
            "num_hof": [0, 0],
            "num_rlf": [1, 0],
        }
    )
    result = build_event_rate_plot_data(metrics)
    assert result["lambda"].tolist() == [0.0, 0.1]
    assert result.loc[result["lambda"] == 0.0, "ho_per_s"].iloc[0] == pytest.approx(0.2)
    points = build_operating_point_plot_data(
        metrics.assign(mean_capacity=[4.0, 3.8], rel_connected_time=[0.9, 0.95])
    )
    assert points["outage"].tolist() == pytest.approx([0.1, 0.05])


def test_outage_stacks_are_exactly_conservative() -> None:
    outage = pd.DataFrame(
        {
            "split": ["id_test"],
            "method": ["a3"],
            "selection_level": ["speed_aware"],
            "lambda": [10.0],
            "successful_ho_exec_seconds": [1.2],
            "hof_related_seconds": [0.4],
            "rlf_no_ho_seconds": [0.8],
            "outage_seconds": [2.4],
        }
    )
    result = build_outage_stacked_plot_data(outage)
    assert result["outage_seconds"].iloc[0] == pytest.approx(2.4)

    outage.loc[0, "outage_seconds"] = 2.5
    with pytest.raises(ValueError, match="do not conserve"):
        build_outage_stacked_plot_data(outage)


def test_figures_render_with_log_axis_thresholds_and_stacks(tmp_path: Path) -> None:
    metrics = pd.DataFrame(
        {
            "split": ["id_test"] * 4,
            "method": ["milp", "milp", "a3", "a3"],
            "selection_level": ["none", "none", "global", "global"],
            "lambda": [0.1, 1.0, 0.1, 1.0],
            "simulated_time_s": [10.0] * 4,
            "num_ho": [1, 2, 1, 1],
            "num_hof": [0, 0, 1, 0],
            "num_rlf": [0, 1, 1, 0],
        }
    )
    plot_event_rates(
        build_event_rate_plot_data(metrics), tmp_path / "events", formats=("png",)
    )
    operating = build_operating_point_plot_data(
        metrics.assign(mean_capacity=[4.0, 3.8, 3.9, 3.7], rel_connected_time=0.9)
    )
    plot_operating_points(operating, tmp_path / "operating", formats=("png",))

    attempts = pd.DataFrame(
        {
            "split": ["id_test", "id_test"],
            "method": ["milp", "a3"],
            "selection_level": ["none", "global"],
            "lambda": [1.0, 1.0],
            "outcome": ["success", "hof_exec"],
            "prep_end_step": [2, 2],
            "exec_end_step": [4, 4],
            "prep_end_l1_qin_sinr_db": [-3.0, -4.0],
            "prep_end_l1_qout_sinr_db": [-7.0, -8.0],
            "exec_end_l1_qin_sinr_db": [-2.0, -5.0],
            "exec_end_l1_qout_sinr_db": [-6.0, -9.0],
            "prep_end_rlm_evaluated": [True, False],
            "exec_end_rlm_evaluated": [True, False],
        }
    )
    plot_handover_sinr_cdfs(
        build_handover_sinr_cdf_data(attempts),
        tmp_path / "cdf",
        q_in_db=-6.0,
        q_out_db=-8.0,
        formats=("png",),
    )

    outage = pd.DataFrame(
        {
            "split": ["id_test", "id_test"],
            "method": ["milp", "a3"],
            "selection_level": ["none", "global"],
            "lambda": [1.0, 1.0],
            "successful_ho_exec_seconds": [1.0, 0.8],
            "hof_related_seconds": [0.0, 0.2],
            "rlf_no_ho_seconds": [0.4, 0.3],
            "outage_seconds": [1.4, 1.3],
        }
    )
    plot_outage_stacked(
        build_outage_stacked_plot_data(outage), tmp_path / "outage", formats=("png",)
    )
    assert (tmp_path / "events.png").is_file()
    assert (tmp_path / "outage.png").is_file()
    assert (tmp_path / "operating" / "rate_outage_frontier.png").is_file()
    assert (tmp_path / "operating" / "rate_outage_frontier_zoom.png").is_file()
    assert [path.name for path in (tmp_path / "cdf").glob("*.png")] == [
        "ho_execution_sinr_cdf.png"
    ]
