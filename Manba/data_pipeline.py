from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import importlib.util
import numpy as np

_BASE_DIR = Path(__file__).resolve().parent
_XLSTM_DIR = _BASE_DIR.parent / "xgboost+lstm"

_cfg_spec = importlib.util.spec_from_file_location("cfg", _XLSTM_DIR / "01_config.py")
cfg = importlib.util.module_from_spec(_cfg_spec)
assert _cfg_spec and _cfg_spec.loader
_cfg_spec.loader.exec_module(cfg)

_dc_spec = importlib.util.spec_from_file_location("dc", _XLSTM_DIR / "02_data_cleaning.py")
dc = importlib.util.module_from_spec(_dc_spec)
assert _dc_spec and _dc_spec.loader
_dc_spec.loader.exec_module(dc)

_fe_spec = importlib.util.spec_from_file_location("fe", _XLSTM_DIR / "03_feature_engineering.py")
fe = importlib.util.module_from_spec(_fe_spec)
assert _fe_spec and _fe_spec.loader
_fe_spec.loader.exec_module(fe)


@dataclass
class TrainingRecord:
    site_id: int
    date_value: date | str | None
    seq: np.ndarray
    tab: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    prev_targets: np.ndarray
    current_targets: np.ndarray


def load_site_rows() -> dict[int, list[dict[str, Any]]]:
    survey_path, weather_path = dc.unzip_inputs()
    metadata, _ = dc.read_station_metadata(weather_path)
    weather_by_site = dc.read_weather_series(weather_path, metadata)
    panel_rows, _ = dc.read_and_aggregate_survey(survey_path, weather_by_site)
    return fe.add_process_features(panel_rows, weather_by_site)


def build_training_records(
    site_rows: dict[int, list[dict[str, Any]]],
    site_ids: list[int],
    targets: list[str],
) -> list[TrainingRecord]:
    records: list[TrainingRecord] = []
    for site_id in site_ids:
        rows = site_rows[site_id]
        sub_sequences = fe.split_rows_by_replicate(rows)

        for subset in sub_sequences:
            if len(subset) < 2:
                continue
            for idx in range(1, len(subset)):
                current = subset[idx]
                previous = subset[idx - 1]

                prev_targets = np.asarray(
                    [
                        fe.fill_none(previous.get(targets[0])) / 100.0,
                        fe.fill_none(previous.get(targets[1])) / 100.0,
                    ],
                    dtype=np.float32,
                )
                current_targets = np.asarray(
                    [
                        fe.fill_none(current.get(targets[0])) / 100.0,
                        fe.fill_none(current.get(targets[1])) / 100.0,
                    ],
                    dtype=np.float32,
                )

                seq = np.asarray(current["weather_seq_28"], dtype=np.float32)
                tab_values = [fe.fill_none(current.get(name)) for name in cfg.BASE_MODEL_FEATURES]
                tab_values.extend(prev_targets.tolist())
                tab = np.asarray(tab_values, dtype=np.float32)

                y = current_targets - prev_targets
                mask = np.asarray(
                    [
                        1.0 if current.get(targets[0]) is not None else 0.0,
                        1.0 if current.get(targets[1]) is not None else 0.0,
                    ],
                    dtype=np.float32,
                )

                date_value = current.get("date") or current.get("date_str")

                records.append(
                    TrainingRecord(
                        site_id=site_id,
                        date_value=date_value,
                        seq=seq,
                        tab=tab,
                        y=y.astype(np.float32),
                        mask=mask,
                        prev_targets=prev_targets,
                        current_targets=current_targets,
                    )
                )

    return records
