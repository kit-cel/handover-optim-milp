"""Module for storing simulation results."""

import csv
import os
from typing import Any
import warnings

from ..common.base_config import BaseConfig


class ResultsMetrics:
    """Class for storing result metrics."""

    meta: dict[str, Any]
    result_metrics: dict[int, dict[str, Any]] | None
    agg_result_metrics: dict[str, Any] | None

    def __init__(self, config: BaseConfig, ep_idx: int) -> None:
        """
        Initialize the OptimizationResult instance.

        Parameters
        ----------
        config : OptimConfig
            Optimization configuration used for this result.
        ep_idx : int
            Episode index.
        """
        self.config = config
        self.ep_idx = ep_idx

        self.meta = {}
        self.result_metrics = None
        self.agg_result_metrics = None

    def add_meta(self, meta_dict: dict[str, Any]) -> None:
        """Add additional metadata."""
        self.meta.update(meta_dict)

    def add_result_metrics(self, result_metrics: dict[int, dict[str, Any]]) -> None:
        """Add optimization result metrics."""
        self.result_metrics = result_metrics

    def add_aggregated_result_metrics(self, agg_result_metrics: dict[str, Any]) -> None:
        """Add optimization result metrics."""
        self.agg_result_metrics = agg_result_metrics

    def save_metrics_to_csv(
        self,
        *,
        filename: str | None = None,
        subfolder: str | None = None,
        filename_include_meta: list[str] | None = None,
    ) -> None:
        """Save results and metadata to a CSV file."""
        if self.result_metrics is None:
            raise ValueError("Result metrics must be set before saving to CSV.")

        if filename is None:
            filename = self._build_filename(filename_include_meta)

        log_dir = os.path.join(self.config.base_dir, "results")
        if subfolder is not None:
            log_dir = os.path.join(log_dir, subfolder)
        os.makedirs(log_dir, exist_ok=True)

        csv_path = os.path.join(log_dir, filename)
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._get_fieldnames(), delimiter=";")

            if not file_exists:
                writer.writeheader()

            for ue_idx, ue_metrics in self.result_metrics.items():
                row = self._join_meta_and_metric_dict(self.meta, ue_metrics)
                if "ue_idx" not in row:
                    row["ue_idx"] = ue_idx
                writer.writerow(row)

    def get_agg_metrics_with_meta(self) -> dict[str, Any]:
        """Get a dictionary of aggregated metrics combined with meta information."""
        if self.agg_result_metrics is None:
            raise ValueError("Aggregated result metrics not set.")
        return self._join_meta_and_metric_dict(self.meta, self.agg_result_metrics)

    def _get_fieldnames(self) -> list[str]:
        """Get fieldnames for CSV based on meta and result metrics."""
        if self.result_metrics is None:
            raise ValueError("Result metrics must be set to determine fieldnames.")

        fieldnames = set(self.meta.keys())
        for ue_metrics in self.result_metrics.values():
            fieldnames.update(ue_metrics.keys())

        return sorted(fieldnames)

    def _build_filename(self, filename_include_meta: list[str] | None = None) -> str:
        """Build a filename based on included metadata fields."""
        if filename_include_meta is None:
            return f"metrics_ep{self.ep_idx:04d}.csv"

        meta_parts = []
        for key in filename_include_meta:
            if key not in self.meta:
                warnings.warn(
                    f"Metadata key '{key}' specified for filename is not present in meta."
                )
                continue
            value = self.meta.get(key, "NA")
            meta_parts.append(f"{key}{value}")

        if len(meta_parts) == 0:
            return f"metrics_ep{self.ep_idx:04d}.csv"

        meta_str = "_".join(meta_parts)
        return f"metrics_ep{self.ep_idx:04d}_{meta_str}.csv"

    def _join_meta_and_metric_dict(
        self, meta: dict[str, Any], metric_dict: dict[str, Any]
    ) -> dict[Any, Any]:
        """Join meta and metric dictionaries, checking for duplicate keys."""
        duplicate_keys = set(meta.keys()) & set(metric_dict.keys())
        if duplicate_keys:
            raise ValueError(
                f"Duplicate keys found between meta and result_metrics: {duplicate_keys}"
            )
        return {**meta, **metric_dict}
