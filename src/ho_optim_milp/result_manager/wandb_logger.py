"""This module provides a class to log experiments using Weights & Biases (wandb)."""

import os
import subprocess
from typing import TYPE_CHECKING

import wandb

if TYPE_CHECKING:
    from ..common.base_config import BaseConfig


class WandBLogger:
    """A class to log experiments using Weights & Biases (wandb), enriched with Git metadata.

    This logger automatically attaches Git commit info, repo cleanliness status, and repository
    URL (as HTTPS) to each W&B run. It also optionally uploads the configuration as an artifact.

    Parameters
    ----------
    project : str
        The name of the W&B project.
    name : str | None, optional
        A name for the W&B run (default is None).
    config : SimulationConfig, optional
        Configuration object containing experiment parameters (default is None).
    tags : list[str] | None, optional
        Additional tags to label the run (default is None).
    """

    def __init__(
        self,
        config: "BaseConfig",
        *,
        project: str | None = None,
        run_id: str | None = None,
        run_name: str | None = None,
        run_type: str | None = None,
        team_name: str | None = None,
        group_name: str | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
        enable: bool = True,
    ):
        """Initialize the WandBLogger."""
        if not enable:
            self.run = None
            return

        self.git_sha = self._get_git_commit_sha()
        self.git_dirty = self._check_git_dirty()
        self.git_repo_url = self._get_git_repo_url()

        self.log_dir = os.path.join(config.base_dir, config.log_dir)

        full_tags = tags or []
        if self.git_sha:
            full_tags.append(f"git:{self.git_sha}")
        if self.git_dirty:
            full_tags.append("git:dirty")

        self.run = wandb.init(
            entity=team_name,
            project=project,
            dir=self.log_dir,
            id=run_id,
            name=run_name,
            job_type=run_type,
            group=group_name,
            notes=notes,
            tags=full_tags,
            config=config.to_dict(),
            mode="online",
            save_code=False,
        )

        if self.git_sha:
            wandb.config.update({"git/commit": self.git_sha}, allow_val_change=True)
        if self.git_repo_url:
            wandb.config.update(
                {"git/repo_url": self.git_repo_url}, allow_val_change=True
            )
        wandb.config.update({"git/dirty": self.git_dirty}, allow_val_change=True)

        self._print_git_status()

    def log(self, data: dict, step: int | None = None) -> None:
        """Log data to the current W&B run."""
        if self.run is None:
            return
        wandb.log(data, step=step)

    def finish_run(self) -> None:
        """Finish the current W&B run."""
        if self.run is not None:
            self.run.finish()

    def _get_git_commit_sha(self) -> str | None:
        """Return the current Git commit SHA (short form), or None if unavailable."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
                text=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def _check_git_dirty(self) -> bool:
        """Check whether the Git repository has uncommitted changes.

        Returns
        -------
        bool
            True if uncommitted changes exist, False otherwise.
        """
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
                text=True,
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False

    def _get_git_repo_url(self) -> str | None:
        """Get the remote URL of the Git repository.

        Returns
        -------
        str | None
            The remote repository URL, or None if not available.
        """
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
                text=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def _convert_to_https(self, url: str) -> str:
        """Convert an SSH-style Git URL to an HTTPS URL for clickability in terminals.

        Parameters
        ----------
        url : str
            The original Git URL (e.g., SSH format).

        Returns
        -------
        str
            A converted HTTPS URL suitable for clickable display.
        """
        if url.startswith("git@"):
            url = url.replace("git@", "")
            url = url.replace(":", "/")
            return f"https://{url}".removesuffix(".git")
        return url

    def _print_git_status(self):
        """Print Git commit, repo URL, and cleanliness status to the terminal."""
        if self.git_sha:
            print(f"Commit SHA: {self.git_sha}")
        if self.git_repo_url:
            https_url = self._convert_to_https(self.git_repo_url)
            print(f"Repo URL: {https_url}")
        if self.git_dirty:
            print("Warning: Repository has uncommitted changes.")
        else:
            print("Repo clean.")
