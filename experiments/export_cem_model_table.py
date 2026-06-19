from __future__ import annotations
'''
python experiments/export_cem
_model_table.py outputs/models/hotel_joint_agent_20260509_203212.json
/Users/raily/Desktop/hotel_pricing/ABM_hotel_pricing/outputs/models/hotel_joint_agent_20260509_203212_decoded_full.csv
'''
import argparse
import csv
import json
import math
from pathlib import Path


BUCKET_LABELS = ["0", "1", "2-3", "4-6", "7-13", "14-29", "30-59", "60-90"]
INVENTORY_NAMES = ["very_low_inv", "low_inv", "mid_inv", "high_inv", "very_high_inv"]
SEASON_NAMES = ["low", "mid", "high"]
WEEKDAY_NAMES = ["workday", "weekend"]
DEFAULT_MEAN = [110.0, 110.0]
DEFAULT_COV = [[2500.0, 0.0], [0.0, 2500.0]]
N_INVENTORY_LEVELS = 5
N_SEASONS = 3
N_WEEKDAY_TYPES = 2
N_STAGE_BUCKETS = 8
BASE_STATE_COUNT = N_INVENTORY_LEVELS * N_SEASONS * N_WEEKDAY_TYPES
TOTAL_STATES = BASE_STATE_COUNT * N_STAGE_BUCKETS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode a saved CEM model into a full 240-state CSV table.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def decode_state(state_id: int) -> tuple[int, int, int, int]:
    inventory_level = state_id // (N_SEASONS * N_WEEKDAY_TYPES * N_STAGE_BUCKETS)
    season = (state_id // (N_WEEKDAY_TYPES * N_STAGE_BUCKETS)) % N_SEASONS
    weekday = (state_id // N_STAGE_BUCKETS) % N_WEEKDAY_TYPES
    stage_id = state_id % N_STAGE_BUCKETS
    return inventory_level, season, weekday, stage_id


def learning_status(
    present_in_model: bool,
    is_default_mean: bool,
    is_default_cov: bool,
    visit_count: int,
    min_learn_visits: int,
) -> str:
    if not present_in_model:
        return "missing"
    if is_default_mean and is_default_cov:
        return "default_unlearned"
    if visit_count < min_learn_visits:
        return "learned_sparse"
    return "learned"


def export_table(model_path: Path, output_path: Path) -> Path:
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    means = {int(k): v for k, v in payload.get("means", {}).items()}
    covs = {int(k): v for k, v in payload.get("covs", {}).items()}
    visits = {int(k): int(v) for k, v in payload.get("state_visit_count", {}).items()}
    min_learn_visits = int(payload.get("n_elite", 120))

    fieldnames = [
        "state_id",
        "present_in_model",
        "missing_in_model",
        "visit_count",
        "inventory_level_code",
        "inventory_level_name",
        "season_code",
        "season_name",
        "weekday_code",
        "weekday_name",
        "stage_id",
        "bucket_label",
        "mean_online_base",
        "mean_offline",
        "std_online",
        "std_offline",
        "cov_01",
        "cov_10",
        "is_default_mean",
        "is_default_cov",
        "learning_status",
    ]

    rows = []
    for state_id in range(TOTAL_STATES):
        inventory_level, season, weekday, stage_id = decode_state(state_id)
        present_in_model = state_id in means or state_id in covs or state_id in visits
        mean = means.get(state_id, [None, None])
        cov = covs.get(state_id, [[None, None], [None, None]])
        visit_count = visits.get(state_id, 0)
        std_online = math.sqrt(cov[0][0]) if cov[0][0] is not None else None
        std_offline = math.sqrt(cov[1][1]) if cov[1][1] is not None else None
        is_default_mean = mean == DEFAULT_MEAN
        is_default_cov = cov == DEFAULT_COV

        rows.append(
            {
                "state_id": state_id,
                "present_in_model": present_in_model,
                "missing_in_model": not present_in_model,
                "visit_count": visit_count,
                "inventory_level_code": inventory_level,
                "inventory_level_name": INVENTORY_NAMES[inventory_level],
                "season_code": season,
                "season_name": SEASON_NAMES[season],
                "weekday_code": weekday,
                "weekday_name": WEEKDAY_NAMES[weekday],
                "stage_id": stage_id,
                "bucket_label": BUCKET_LABELS[stage_id],
                "mean_online_base": mean[0],
                "mean_offline": mean[1],
                "std_online": std_online,
                "std_offline": std_offline,
                "cov_01": cov[0][1],
                "cov_10": cov[1][0],
                "is_default_mean": is_default_mean,
                "is_default_cov": is_default_cov,
                "learning_status": learning_status(
                    present_in_model=present_in_model,
                    is_default_mean=is_default_mean,
                    is_default_cov=is_default_cov,
                    visit_count=visit_count,
                    min_learn_visits=min_learn_visits,
                ),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    model_path = args.model_path.resolve()
    output_path = args.output.resolve() if args.output else model_path.with_name(f"{model_path.stem}_decoded_full.csv")
    saved_path = export_table(model_path=model_path, output_path=output_path)
    print(saved_path)


if __name__ == "__main__":
    main()
