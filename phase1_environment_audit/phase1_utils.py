from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import ABM_CONFIG, ABM_PERTURBATION_TEMPLATE, ENV_CONFIG, RL_CONFIG
from configs.experiment2 import Experiment2Config


ARTIFACTS_DIR = PROJECT_ROOT / "phase1_environment_audit" / "artifacts"
DIAGNOSTICS_DIR = ARTIFACTS_DIR / "diagnostics"
FIXED_PRICE_SCAN_DIR = ARTIFACTS_DIR / "fixed_price_scan"
CUSTOMER_UTILITY_TRACE_DIR = ARTIFACTS_DIR / "customer_utility_trace"
CONFIG_SNAPSHOT_DIR = ARTIFACTS_DIR / "config_snapshots"


def ensure_phase1_dirs() -> None:
    for path in [ARTIFACTS_DIR, DIAGNOSTICS_DIR, FIXED_PRICE_SCAN_DIR, CUSTOMER_UTILITY_TRACE_DIR, CONFIG_SNAPSHOT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_city_hotel_data() -> pd.DataFrame:
    path = PROJECT_ROOT / "datasets" / "hotel_bookings.csv"
    df = pd.read_csv(path)
    return df[df["hotel"] == "City Hotel"].copy()


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def build_runtime_snapshot(config: Experiment2Config) -> dict[str, Any]:
    return {
        "project_root": str(PROJECT_ROOT),
        "abm_perturbation_template": ABM_PERTURBATION_TEMPLATE,
        "experiment2_config": _to_jsonable(config),
        "abm_config": _to_jsonable(ABM_CONFIG),
        "rl_config": _to_jsonable(RL_CONFIG),
        "env_config": _to_jsonable(ENV_CONFIG),
    }


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
