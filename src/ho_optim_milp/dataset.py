"""Read and validate a completed handover dataset.

The reader works entirely from saved artifacts. It never starts a solver or
imports the network simulator. Binary rank-one results are stored by episode;
the unrestricted MILP results remain split-level Parquet files in schema 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml


DATASET_SCHEMA_VERSION = 3
EPISODE_MARKER_SCHEMA = "binary-milp-dataset-v3"
_METRIC_IDENTITY = ("episode_idx", "ue_id", "lambda")


def _safe_path(root: Path, relative_name: str) -> Path:
    relative = PurePosixPath(relative_name)
    if (not relative_name or relative.is_absolute() or ".." in relative.parts
            or "\\" in relative_name or ":" in relative_name):
        raise ValueError(f"Unsafe dataset path: {relative_name}")
    return root.joinpath(*relative.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _require_columns(frame: pd.DataFrame, names: tuple[str, ...], path: Path) -> None:
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")


def validate_dataset(
    root: str | Path, *, verify_artifact_hashes: bool = False
) -> dict[str, Any]:
    """Validate release structure and optionally hash every data artifact.

    The fast check reads only metadata and file sizes. Full hashing reads the
    complete dataset and is intentionally an explicit option.
    """
    directory = Path(root).expanduser().resolve()
    marker_path = directory / "release_complete.json"
    manifest_path = directory / "dataset_manifest.json"
    metadata_path = directory / "dataset.yaml"
    marker = _read_json(marker_path)
    manifest = _read_json(manifest_path)
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={DATASET_SCHEMA_VERSION} in {metadata_path}")
    release_id = metadata.get("release", {}).get("release_id")
    if marker.get("passed") is not True or marker.get("release_id") != release_id:
        raise ValueError("Release marker does not match dataset metadata")
    if marker.get("dataset_manifest") != "dataset_manifest.json":
        raise ValueError("Release marker points to an unexpected manifest")
    if _sha256(manifest_path) != marker.get("dataset_manifest_sha256"):
        raise ValueError("Dataset manifest checksum mismatch")
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("Dataset manifest schema does not match the reader")
    variants = metadata.get("milp_variants", {})
    if not isinstance(variants, dict) or not variants:
        raise ValueError("Dataset has no MILP variants")
    for name, spec in variants.items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            raise ValueError("Invalid MILP variant metadata")
        if spec.get("layout") not in {"legacy", "episode_shards"}:
            raise ValueError(f"Unsupported MILP layout for {name}")
        if not isinstance(spec.get("root"), str):
            raise ValueError(f"Missing MILP result root for {name}")
        _safe_path(directory, spec["root"])
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Dataset manifest has no artifacts")
    expected: set[str] = set()
    total_bytes = 0
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("Invalid artifact entry")
        name = artifact.get("path")
        if not isinstance(name, str):
            raise ValueError("Artifact path must be a string")
        if name in expected:
            raise ValueError(f"Unsafe or duplicate artifact path: {name}")
        expected.add(name)
        path = _safe_path(directory, name)
        if not path.is_file() or path.stat().st_size != artifact.get("bytes"):
            raise ValueError(f"Missing or wrongly sized artifact: {path}")
        if verify_artifact_hashes and _sha256(path) != artifact.get("sha256"):
            raise ValueError(f"Artifact checksum mismatch: {path}")
        total_bytes += artifact["bytes"]
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    if actual != expected | {"dataset_manifest.json", "release_complete.json"}:
        raise ValueError("Dataset contains unlisted or missing files")
    return {"release_id": release_id, "artifact_count": len(artifacts), "artifact_bytes": total_bytes,
            "artifact_hashes_checked": verify_artifact_hashes}


@lru_cache(maxsize=128)
def _verified_episode(
    directory: Path, file_signature: tuple[tuple[str, int, int], ...],
    marker_signature: tuple[int, int],
) -> dict[str, Any]:
    del file_signature, marker_signature
    marker = _read_json(directory / "complete.json")
    if marker.get("schema") != EPISODE_MARKER_SCHEMA:
        raise ValueError(f"Invalid binary episode marker: {directory}")
    for filename in ("metrics.parquet", "trajectories.parquet"):
        expected = marker.get("files", {}).get(filename)
        if not isinstance(expected, str) or _sha256(directory / filename) != expected:
            raise ValueError(f"Binary episode checksum mismatch: {directory / filename}")
    return marker


@dataclass(frozen=True)
class Dataset:
    """Read-only access to a validated dataset root."""

    root: Path
    metadata: dict[str, Any]

    @classmethod
    def open(cls, root: str | Path, *, verify_artifact_hashes: bool = False) -> "Dataset":
        directory = Path(root).expanduser().resolve()
        validate_dataset(directory, verify_artifact_hashes=verify_artifact_hashes)
        metadata = yaml.safe_load((directory / "dataset.yaml").read_text(encoding="utf-8"))
        return cls(directory, metadata)

    def split(self, name: str) -> "DatasetSplit":
        names = {item["name"] for item in self.metadata["release"]["splits"]}
        if name not in names:
            raise KeyError(f"Unknown dataset split: {name}")
        return DatasetSplit(self, name)

    def artifact_path(self, name: str, *, split: str, level: str = "") -> Path:
        """Resolve one declared artifact template inside the dataset root."""
        templates = self.metadata.get("artifacts", {})
        template = templates.get(name) if isinstance(templates, dict) else None
        if not isinstance(template, str):
            raise KeyError(f"Dataset does not declare artifact {name!r}")
        relative = template.replace("<split>", split).replace("<level>", level)
        if "<" in relative or ">" in relative or "*" in relative:
            raise ValueError(f"Unresolved artifact template: {template}")
        return _safe_path(self.root, relative)

    def variant(self, name: str) -> dict[str, Any]:
        """Return the declared storage layout for one MILP variant."""
        variants = self.metadata["milp_variants"]
        if name not in variants:
            raise ValueError(f"Unknown MILP variant: {name}")
        return variants[name]


@dataclass(frozen=True)
class DatasetSplit:
    """Access saved network data and MILP results for one split."""

    dataset: Dataset
    name: str

    @property
    def network_data(self) -> Path:
        path = self.dataset.artifact_path("network_data", split=self.name)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _binary_episode(self, episode_idx: int, variant: str) -> tuple[Path, dict[str, Any]]:
        root = self.dataset.variant(variant)["root"]
        directory = (_safe_path(self.dataset.root, root) / self.name
                     / "episodes" / f"ep_{episode_idx:04d}")
        marker_path = directory / "complete.json"
        if not marker_path.is_file():
            raise ValueError(f"Binary rank-one episode is not published: {directory}")
        signature = tuple(
            (filename, (directory / filename).stat().st_size,
             (directory / filename).stat().st_mtime_ns)
            for filename in ("metrics.parquet", "trajectories.parquet")
        )
        marker_stat = marker_path.stat()
        marker = _verified_episode(
            directory, signature, (marker_stat.st_size, marker_stat.st_mtime_ns)
        )
        if marker.get("episode_idx") != episode_idx or marker.get("split") != self.name:
            raise ValueError(f"Binary episode identity mismatch: {directory}")
        return directory, marker

    def milp_metrics(self, variant: str = "unrestricted") -> pd.DataFrame:
        spec = self.dataset.variant(variant)
        if spec["layout"] == "legacy":
            path = _safe_path(self.dataset.root, spec["root"]) / self.name / "optim_result_metrics.parquet"
            frame = pd.read_parquet(path)
        elif spec["layout"] == "episode_shards":
            published = spec["published_episodes"]
            episode_indices = published.get(self.name, [])
            if not episode_indices:
                raise ValueError(f"No binary rank-one episodes published for {self.name}")
            paths = [self._binary_episode(int(index), variant)[0] / "metrics.parquet" for index in episode_indices]
            path = paths[0]
            frame = pd.concat((pd.read_parquet(item) for item in paths), ignore_index=True)
        _require_columns(frame, _METRIC_IDENTITY + ("mean_capacity", "num_ho", "num_rlf"), path)
        return frame

    def milp_trajectory(
        self, *, episode_idx: int, ue_id: int, lambda_value: float,
        variant: str = "unrestricted",
    ) -> pd.DataFrame:
        spec = self.dataset.variant(variant)
        if spec["layout"] == "legacy":
            path = _safe_path(self.dataset.root, spec["root"]) / self.name / "optim_results.parquet"
            frame = pd.read_parquet(path, filters=[
                ("episode_idx", "==", episode_idx), ("ue_id", "==", ue_id),
                ("lambda", "==", lambda_value),
            ])
        elif spec["layout"] == "episode_shards":
            directory, marker = self._binary_episode(episode_idx, variant)
            metrics_path = directory / "metrics.parquet"
            metrics = pd.read_parquet(metrics_path)
            _require_columns(metrics, _METRIC_IDENTITY, metrics_path)
            matches = metrics.index[
                metrics["episode_idx"].eq(episode_idx)
                & metrics["ue_id"].eq(ue_id)
                & metrics["lambda"].eq(lambda_value)
            ].tolist()
            if len(matches) != 1:
                raise ValueError(f"Expected one binary trajectory identity, found {len(matches)}")
            path = directory / "trajectories.parquet"
            with pq.ParquetFile(path) as parquet:
                if parquet.num_row_groups != len(metrics) or marker.get("row_groups") != len(metrics):
                    raise ValueError(f"Binary episode row-group mismatch: {directory}")
                frame = parquet.read_row_group(matches[0]).to_pandas()
        if frame.empty:
            raise ValueError(f"MILP trajectory is missing in {path}")
        _require_columns(frame, ("time_step", "pcell"), path)
        if spec["layout"] == "legacy":
            _require_columns(frame, _METRIC_IDENTITY, path)
            identities = frame.loc[:, _METRIC_IDENTITY].drop_duplicates()
            if len(identities) != 1 or tuple(identities.iloc[0]) != (episode_idx, ue_id, lambda_value):
                raise ValueError(f"MILP trajectory identity mismatch in {path}")
        return frame.sort_values("time_step").reset_index(drop=True)

    def a3_metrics(self, level: str = "global") -> pd.DataFrame:
        """Load one stored A3 selection policy's metrics for this split."""
        if level not in {"global", "speed_aware", "split_oracle"}:
            raise ValueError(f"Unknown A3 selection level: {level}")
        path = self.dataset.artifact_path("a3_metrics", split=self.name, level=level)
        frame = pd.read_parquet(path)
        _require_columns(frame, _METRIC_IDENTITY + ("mean_capacity", "num_ho", "num_rlf"), path)
        return frame

    def a3_trajectory(
        self, *, episode_idx: int, ue_id: int, lambda_value: float,
        level: str = "global",
    ) -> pd.DataFrame:
        """Read one A3 trajectory without materializing the whole split."""
        if level not in {"global", "speed_aware", "split_oracle"}:
            raise ValueError(f"Unknown A3 selection level: {level}")
        path = self.dataset.artifact_path("a3_trajectories", split=self.name, level=level)
        frame = pd.read_parquet(path, filters=[
            ("episode_idx", "==", episode_idx), ("ue_id", "==", ue_id),
            ("lambda", "==", lambda_value),
        ])
        if frame.empty:
            raise ValueError(f"A3 trajectory is missing in {path}")
        _require_columns(frame, _METRIC_IDENTITY + ("time_step", "pcell"), path)
        if len(frame.loc[:, _METRIC_IDENTITY].drop_duplicates()) != 1:
            raise ValueError(f"A3 trajectory identity mismatch in {path}")
        return frame.sort_values("time_step").reset_index(drop=True)

    def no_handover_metrics(self) -> pd.DataFrame:
        """Load the saved no-handover baseline metrics."""
        path = self.dataset.artifact_path("no_ho_metrics", split=self.name)
        frame = pd.read_parquet(path)
        _require_columns(frame, _METRIC_IDENTITY + ("mean_capacity", "num_ho", "num_rlf"), path)
        return frame

    def events(self, table: str) -> pd.DataFrame:
        """Load one derived event table for this split."""
        if table not in {"rrc_events", "handover_attempts", "outage_decomposition"}:
            raise ValueError(f"Unknown event table: {table}")
        path = self.dataset.artifact_path(table, split=self.name)
        return pd.read_parquet(path)

    def figures(self) -> tuple[Path, ...]:
        """List existing figure files for this split."""
        directory = self.dataset.root / "plots" / self.name
        return tuple(sorted(path for path in directory.iterdir() if path.suffix in {".pdf", ".png"}))
