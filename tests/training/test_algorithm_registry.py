from __future__ import annotations

from src.training.algorithm_registry import get_algorithm_choices, get_algorithm_runner


def test_off_policy_algorithms_are_registered() -> None:
    choices = set(get_algorithm_choices())

    assert {"sac", "td3", "tqc"}.issubset(choices)
    assert get_algorithm_runner("td3")["config"].save_name == "td3_idea2_hotel"
    assert get_algorithm_runner("tqc")["config"].save_name == "tqc_idea2_hotel"
