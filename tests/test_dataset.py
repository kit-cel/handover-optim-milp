"""Fast schema-3 integrity and episode-reader regressions."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ho_optim_milp.dataset import Dataset, validate_dataset


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release(root: Path) -> Path:
    binary = root / "optim_results_binary_rank1/id_val/episodes/ep_0000"
    binary.mkdir(parents=True)
    (root / "network_data").mkdir()
    (root / "network_data/id_val.h5").write_bytes(b"saved radio trace")
    metrics = pd.DataFrame({
        "episode_idx": [0], "ue_id": [2], "lambda": [1.0],
        "mean_capacity": [3.0], "num_ho": [0], "num_rlf": [0],
    })
    metrics.to_parquet(binary / "metrics.parquet", index=False)
    pd.DataFrame({"time_step": [0, 1, 2], "pcell": [1, 1, 1]}).to_parquet(
        binary / "trajectories.parquet", index=False
    )
    (binary / "complete.json").write_text(json.dumps({
        "schema": "binary-milp-dataset-v3", "split": "id_val", "episode_idx": 0,
        "rows": 1, "steps": 3, "row_groups": 1,
        "files": {name: _hash(binary / name) for name in
                  ("metrics.parquet", "trajectories.parquet")},
    }), encoding="utf-8")
    (root / "dataset.yaml").write_text(yaml.safe_dump({
        "schema_version": 3,
        "release": {"release_id": root.name, "splits": [{"name": "id_val"}]},
        "milp_variants": {
            "unrestricted": {"layout": "legacy", "root": "optim_results"},
            "binary_rank1": {
                "layout": "episode_shards", "root": "optim_results_binary_rank1",
                "published_episodes": {"id_val": [0]}, "complete": True,
            },
        },
        "artifacts": {"network_data": "network_data/<split>.h5"},
    }), encoding="utf-8")
    artifacts = [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
         "sha256": _hash(path)}
        for path in sorted(root.rglob("*")) if path.is_file()
    ]
    (root / "dataset_manifest.json").write_text(json.dumps({
        "schema_version": 3, "artifacts": artifacts,
    }), encoding="utf-8")
    (root / "release_complete.json").write_text(json.dumps({
        "passed": True, "release_id": root.name,
        "dataset_manifest": "dataset_manifest.json",
        "dataset_manifest_sha256": _hash(root / "dataset_manifest.json"),
    }), encoding="utf-8")
    return root


def test_binary_episode_reader_and_integrity(tmp_path: Path) -> None:
    root = _release(tmp_path / "sample_release")
    assert validate_dataset(root, verify_artifact_hashes=True)["artifact_count"] == 5
    split = Dataset.open(root).split("id_val")
    assert len(split.milp_metrics("binary_rank1")) == 1
    steps = split.milp_trajectory(
        episode_idx=0, ue_id=2, lambda_value=1.0, variant="binary_rank1"
    )
    assert steps["pcell"].tolist() == [1, 1, 1]
    with pytest.raises(ValueError, match="Expected one binary trajectory"):
        split.milp_trajectory(
            episode_idx=0, ue_id=3, lambda_value=1.0, variant="binary_rank1"
        )


def test_milp_variant_name_comes_from_dataset_metadata(tmp_path: Path) -> None:
    root = _release(tmp_path / "sample_release")
    metadata_path = root / "dataset.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["milp_variants"]["candidate_limited"] = metadata["milp_variants"].pop(
        "binary_rank1"
    )
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        if artifact["path"] == "dataset.yaml":
            artifact["bytes"] = metadata_path.stat().st_size
            artifact["sha256"] = _hash(metadata_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    marker_path = root / "release_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["dataset_manifest_sha256"] = _hash(manifest_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    split = Dataset.open(root).split("id_val")
    assert len(split.milp_metrics("candidate_limited")) == 1


def test_corrupt_episode_is_rejected(tmp_path: Path) -> None:
    root = _release(tmp_path / "sample_release")
    path = root / "optim_results_binary_rank1/id_val/episodes/ep_0000/metrics.parquet"
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="wrongly sized artifact"):
        Dataset.open(root)


def test_unsupported_schema_is_rejected(tmp_path: Path) -> None:
    root = _release(tmp_path / "sample_release")
    metadata = yaml.safe_load((root / "dataset.yaml").read_text())
    metadata["schema_version"] = 2
    (root / "dataset.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version=3"):
        Dataset.open(root)
