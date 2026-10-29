"""Small checks for the recorded coarse-to-fine A3 search."""

from ho_optim_milp.dataset_generator.a3_protocol import A3SearchProtocol
from ho_optim_milp.dataset_generator.recorded_steps import _a3_profile, _parameters

from test_model_profiles import make_config


def test_refinement_uses_only_selection_split() -> None:
    protocol = A3SearchProtocol(
        coarse_ttt_ms=(0.0, 40.0), coarse_hys_db=(0.0,),
        coarse_offset_db=(0.0,), fine_ttt_ms=(80.0,),
        fine_hys_db=(0.0,), fine_offset_db=(0.0,), top_k=1,
        ttt_neighbor_radius=0, hys_neighbor_radius=0,
        offset_neighbor_radius=0,
    )
    rows = [
        {"method": "a3", "split": "id_val", "lambda": 1.0,
         "ttt_ms": ttt, "hys_db": 0.0, "offset_db": 0.0,
         "mean_capacity": rate, "rel_connected_time": 1.0}
        for ttt, rate in ((0.0, 1.0), (40.0, 2.0))
    ]
    rows.append({**rows[0], "split": "id_test", "mean_capacity": 100.0})
    grids = protocol.refinement_grids(
        rows, lambda_values=[1.0], selection_split="id_val"
    )
    assert grids == {1.0: ((80.0, 0.0, 0.0),)}
    profile = _a3_profile(make_config().rrc, grids[1.0][0])
    assert _parameters(profile) == (80.0, 0.0, 0.0)
