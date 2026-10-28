"""Recorded-data compatibility and standalone publication checks."""

import json
from pathlib import Path

import h5py
import gurobipy as gp
import numpy as np
import pandas as pd
import pytest

from ho_optim_milp import dataset_pipeline
from ho_optim_milp.optimization.optim_config import OptimConfig
from ho_optim_milp.reference.rrc_config import RRCConfig
from ho_optim_milp.result_manager.network_simulation_result import NetworkSimulationResult


def make_network_file(path: Path, *, schema: str) -> None:
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("meta")
        meta.create_dataset("config_common", data=json.dumps({"t_res_ms": 40}))
        if schema == "paper":
            meta.create_dataset("included_keys", data=json.dumps([
                "rsrp", "sinr", "ue_pos", "distances_2d", "distances_3d", "ue_move_angle"
            ]))
        else:
            meta.create_dataset("release", data=json.dumps({"schema_version": 2}))
        episode = handle.create_group("ep_0000")
        sinr = np.full((8, 1, 2), 10.0, dtype=np.float32)
        sinr[:, :, 1] = 0.0
        episode.create_dataset("sinr", data=sinr)
        episode.create_dataset("rsrp", data=sinr - 70.0)
        episode.create_dataset("ue_pos", data=np.zeros((8, 1, 2), dtype=np.float32))
        episode.create_dataset("distances_2d", data=np.ones((8, 1, 2), dtype=np.float32))
        episode.create_dataset("distances_3d", data=np.ones((8, 1, 2), dtype=np.float32))
        episode.create_dataset("ue_move_angle", data=np.zeros((8, 1), dtype=np.float32))
        group = episode.create_group("meta")
        group.create_dataset("config_full", data=json.dumps({"t_res_ms": 40}))


@pytest.mark.parametrize("schema", ["paper", "current"])
def test_both_recorded_hdf5_schemas_load(tmp_path: Path, schema: str) -> None:
    path = tmp_path / "network.h5"
    make_network_file(path, schema=schema)
    result = NetworkSimulationResult.load(
        str(path), keys=["rsrp", "sinr", "ue_pos", "distances_2d", "distances_3d",
                         "ue_move_angle"], validate_data=True
    )
    assert result.n_eps_loaded == 1
    assert result.get_ep_result_by_idx(0).config_common["t_res_ms"] == 40


def test_standalone_dataset_build_has_metrics_figures_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.h5"
    make_network_file(source, schema="current")
    rrc = RRCConfig(t_res_ms=40, n310=1, n311=10, t310_ms=120,
                    ho_event_config={"a3": {"ttt_ms": 40, "hys": 0.0, "offset": 1.0}})
    config = OptimConfig(lambda_r=1.0, rrc=rrc)

    def fake_milp(episode: object, ue_idx: int, optim_config: OptimConfig):
        return (
            {"mean_capacity": 2.0, "rel_connected_time": 0.9,
             "num_ho": 1, "num_rlf": 0, "obj_value": 1.9},
            pd.DataFrame({"time_step": range(8), "pcell": [0] * 8}),
        )

    def fake_a3(episode: object, profile: RRCConfig, backend: str):
        return [{"ue_id": 0, "mean_capacity": 1.5,
                 "rel_connected_time": 0.85, "num_ho": 1, "num_rlf": 0}]

    monkeypatch.setattr(dataset_pipeline, "_run_milp", fake_milp)
    monkeypatch.setattr(dataset_pipeline, "_run_a3", fake_a3)
    output = dataset_pipeline.build_dataset(
        {"id_val": source}, config, [0.0, 1.0], tmp_path / "result"
    )
    assert (output / "release_complete.json").is_file()
    assert (output / "plots" / "id_val" /
            "rate_outage_pareto_frontiers_q-90-95-99.png").is_file()
    assert (output / "plots" / "id_val" /
            "rate_connected_time_vs_lambda_q99.png").is_file()
    assert len(pd.read_parquet(output / "optim_results" / "id_val" /
                               "optim_result_metrics.parquet")) == 2
    assert len(pd.read_parquet(output / "reference_results" / "id_val" /
                               "reference_result_metrics.parquet")) == 1
    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]
    assert all("sha256" in item for item in manifest["artifacts"])
    assert dataset_pipeline.validate_dataset(output)["artifact_count"] == len(manifest["artifacts"])

    with h5py.File(output / "network_data" / "id_val.h5", "r+") as handle:
        handle["ep_0000/sinr"][0, 0, 0] = -99.0
    with pytest.raises(ValueError, match="integrity check"):
        dataset_pipeline.validate_dataset(output)


def test_dataset_rejects_missing_a3_scientific_config(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    make_network_file(source, schema="paper")
    config = OptimConfig(lambda_r=1.0, rrc=RRCConfig(t_res_ms=40))
    with pytest.raises(ValueError, match="A3 profile"):
        dataset_pipeline.build_dataset({"train": source}, config, [1.0], tmp_path / "out")


def test_resume_rejects_modified_step_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.h5"
    make_network_file(source, schema="current")
    rrc = RRCConfig(t_res_ms=40,
                    ho_event_config={"a3": {"ttt_ms": 40, "hys": 0.0, "offset": 1.0}})
    config = OptimConfig(lambda_r=1.0, rrc=rrc)
    monkeypatch.setattr(dataset_pipeline, "_run_milp", lambda *_: (
        {"mean_capacity": 2.0, "rel_connected_time": 0.9, "num_ho": 0},
        pd.DataFrame({"time_step": range(8), "pcell": [0] * 8}),
    ))
    monkeypatch.setattr(dataset_pipeline, "_run_a3", lambda *_: [
        {"ue_id": 0, "mean_capacity": 1.5, "rel_connected_time": 0.8}
    ])
    output = dataset_pipeline.build_dataset(
        {"train": source}, config, [1.0], tmp_path / "release"
    )
    (output / "release_complete.json").unlink()
    (output / "dataset_manifest.json").unlink()
    step = next((output / "optim_results" / "train" / "steps").glob("*.parquet"))
    frame = pd.read_parquet(step)
    frame.loc[0, "pcell"] = 1
    frame.to_parquet(step, index=False)
    with pytest.raises(ValueError, match="step result failed integrity check"):
        dataset_pipeline.build_dataset(
            {"train": source}, config, [1.0], output
        )


@pytest.mark.parametrize("schema", ["paper", "current"])
def test_dataset_build_runs_real_standalone_a3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schema: str
) -> None:
    source = tmp_path / "source.h5"
    make_network_file(source, schema=schema)
    rrc = RRCConfig(t_res_ms=40, n310=1, n311=10, t310_ms=120,
                    ho_event_config={"a3": {"ttt_ms": 40, "hys": 0.0, "offset": 1.0}})
    config = OptimConfig(lambda_r=1.0, rrc=rrc)

    def fake_milp(episode: object, ue_idx: int, optim_config: OptimConfig):
        return ({"mean_capacity": 2.0, "rel_connected_time": 0.9,
                 "num_ho": 1, "num_rlf": 0, "obj_value": 1.9},
                pd.DataFrame({"time_step": range(8), "pcell": [0] * 8}))

    monkeypatch.setattr(dataset_pipeline, "_run_milp", fake_milp)
    output = dataset_pipeline.build_dataset(
        {"train": source}, config, [1.0], tmp_path / "release"
    )
    rows = pd.read_parquet(
        output / "reference_results" / "train" / "reference_result_metrics.parquet"
    )
    assert len(rows) == 1
    assert rows.loc[0, "a3_backend"] == "standalone"
    assert rows.loc[0, "simulated_steps"] == 8
    dataset_pipeline.validate_dataset(output)


@pytest.mark.parametrize("schema,profile", [
    ("paper", "paper_v1"), ("current", "current")
])
def test_recorded_network_data_to_complete_dataset_with_real_milp(
    tmp_path: Path, schema: str, profile: str
) -> None:
    try:
        model = gp.Model("license_probe")
        model.dispose()
    except gp.GurobiError as error:
        pytest.skip(f"Gurobi licence unavailable: {error}")
    source = tmp_path / "source.h5"
    make_network_file(source, schema=schema)
    rrc = RRCConfig(t_res_ms=40, n310=1, n311=10, t310_ms=120,
                    ho_event_config={"a3": {"ttt_ms": 40, "hys": 0.0, "offset": 1.0}})
    config = OptimConfig(lambda_r=1.0, rrc=rrc, model_profile=profile,
                         gurobi={"threads": 1, "log_to_console": False})
    output = dataset_pipeline.build_dataset(
        {"train": source}, config, [0.0, 1.0], tmp_path / "complete"
    )
    metrics = pd.read_parquet(
        output / "optim_results" / "train" / "optim_result_metrics.parquet"
    )
    assert len(metrics) == 2
    assert set(metrics["model_profile"]) == {profile}
    assert (metrics["num_ho"] >= 0).all()
    assert dataset_pipeline.validate_dataset(output)["artifact_count"] > 0
