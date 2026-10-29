"""Integration checks for the released simulator dependency."""

from pathlib import Path

from ho_optim_milp.integrations import simulator
from ho_optim_milp.dataloader.episodes import load_episode
from ho_optim_milp.dataset_generator.recorded_results import reference_recorded_episode

from test_model_profiles import make_config, make_episode


def test_simulator_a3_replays_recorded_network_data() -> None:
    rows = reference_recorded_episode(make_episode(), make_config().rrc,
                                      lambda_r=0.0)
    assert len(rows) == 1
    assert rows[0][0]["ue_id"] == 0
    assert rows[0][0]["simulated_steps"] == 8
    assert 0.0 <= rows[0][0]["rel_connected_time"] <= 1.0


def test_simulator_generates_loadable_network_data(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "simulator_radio_rrc.yaml"
    output = simulator.generate_network_data(fixture, tmp_path / "network.h5")
    episode = load_episode(output, 0)
    assert episode.sinr.shape == (20, 2, 21)
    assert episode.rsrp.shape == episode.sinr.shape
    assert episode.config_common["time_resolution_ms"] == 40
