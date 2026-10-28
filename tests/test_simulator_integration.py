"""Optional simulator behavior must never be required by the core package."""

import builtins
from pathlib import Path

import pytest

from ho_optim_milp.integrations import simulator
from ho_optim_milp.dataloader.episodes import load_episode

from test_model_profiles import make_config, make_episode


def test_missing_simulator_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def without_simulator(name: str, *args: object, **kwargs: object):
        if name.startswith("mobile_network_sim"):
            raise ImportError("mobile_network_sim is absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_simulator)
    with pytest.raises(RuntimeError, match=r"install ho_optim_milp\[simulator\]"):
        simulator._simulator_types()


def test_simulator_a3_replays_recorded_network_data() -> None:
    pytest.importorskip("mobile_network_sim")
    rows = simulator.run_a3_reference(make_episode(), make_config().rrc)
    assert len(rows) == 1
    assert rows[0]["ue_id"] == 0
    assert rows[0]["simulated_steps"] == 8
    assert 0.0 <= rows[0]["rel_connected_time"] <= 1.0


def test_optional_simulator_generates_loadable_network_data(tmp_path: Path) -> None:
    pytest.importorskip("mobile_network_sim")
    fixture = Path(__file__).parent / "fixtures" / "simulator_radio_rrc.yaml"
    output = simulator.generate_network_data(fixture, tmp_path / "network.h5")
    episode = load_episode(output, 0)
    assert episode.sinr.shape == (20, 2, 21)
    assert episode.rsrp.shape == episode.sinr.shape
    assert episode.config_common["time_resolution_ms"] == 40
