from __future__ import annotations

# [03.1] ===== 基础库导入 =====
from collections import defaultdict
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import importlib.util

# [03.2] ===== 动态加载配置与清洗模块 =====
_cfg_spec = importlib.util.spec_from_file_location("cfg", Path(__file__).with_name("01_config.py"))
cfg = importlib.util.module_from_spec(_cfg_spec)
assert _cfg_spec and _cfg_spec.loader
_cfg_spec.loader.exec_module(cfg)

_dc_spec = importlib.util.spec_from_file_location("dc", Path(__file__).with_name("02_data_cleaning.py"))
dc = importlib.util.module_from_spec(_dc_spec)
assert _dc_spec and _dc_spec.loader
_dc_spec.loader.exec_module(dc)


# [03.3] ===== 基础辅助函数 =====
def fill_none(value: float | None, default: float = 0.0) -> float:
    return default if value is None else float(value)


def nearest_weather_index(weather_site: dict[str, Any], obs_date: date) -> int:
    date_index = weather_site["date_index"]
    if obs_date in date_index:
        return date_index[obs_date]
    dates = weather_site["dates"]
    for offset in range(1, 4):
        candidate = obs_date - timedelta(days=offset)
        if candidate in date_index:
            return date_index[candidate]
    if obs_date < dates[0]:
        return 0
    return len(dates) - 1


def padded_sequence(matrix: np.ndarray, end_index: int, lookback: int) -> np.ndarray:
    start_index = end_index - lookback + 1
    if start_index >= 0:
        return matrix[start_index : end_index + 1].copy()
    pad_count = -start_index
    first_row = matrix[0:1].repeat(pad_count, axis=0)
    return np.concatenate([first_row, matrix[0 : end_index + 1]], axis=0)


# [03.4] ===== 过程特征构建 =====
def add_process_features(panel_rows: list[dict[str, Any]], weather_by_site: dict[int, dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    raw_only = bool(getattr(cfg, "USE_RAW_ONLY_FEATURES", False))
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in panel_rows:
        grouped[row["site_id"]].append(deepcopy(row))
    for site_id, rows in grouped.items():
        # rows.sort(key=lambda item: item["date"])
        rows.sort(key=lambda item: (item["date"], item.get("replicate_id_same_day", 0), item.get("record_id", 0)))
        weather_site = weather_by_site[site_id]
        # first_date = rows[0]["date"]

        temp_thresholds = weather_site["temp_thresholds"]

        for position, row in enumerate(rows):
            current_idx = nearest_weather_index(weather_site, row["date"])

            for feature_name in cfg.SEQ_FEATURES:
                if feature_name in weather_site["arrays"]:
                    row[feature_name] = float(weather_site["arrays"][feature_name][current_idx])

            row["weather_seq_28"] = padded_sequence(weather_site["matrix"], current_idx, cfg.LOOKBACK_DAYS)

            if raw_only:
                continue

            window_3 = slice(max(0, current_idx - 2), current_idx + 1)
            window_7 = slice(max(0, current_idx - 6), current_idx + 1)
            window_14 = slice(max(0, current_idx - 13), current_idx + 1)
            window_21 = slice(max(0, current_idx - 20), current_idx + 1)
            window_28 = slice(max(0, current_idx - 27), current_idx + 1)
            cumulative = slice(0, current_idx + 1)
            arrays = weather_site["arrays"]
            row["gdd_cum"] = float(np.maximum(arrays["temp_avg_c"][cumulative] - 10.0, 0.0).sum())
            # row["rain_3d_sum"] = float(arrays["precip_sum"][window_3].sum())
            row["rain_7d_sum"] = float(arrays["precip_sum"][window_7].sum())
            row["rain_14d_sum"] = float(arrays["precip_sum"][window_14].sum())
            row["rain_21d_sum"] = float(arrays["precip_sum"][window_21].sum())
            row["rain_28d_sum"] = float(arrays["precip_sum"][window_28].sum())
            row["rainy_streak_days"] = float(arrays["rainy_streak_days"][current_idx])
            row["rain_gap_days"] = float(arrays["rain_gap_days"][current_idx])
            # row["temp_3d_mean"] = float(arrays["temp_avg_c"][window_3].mean())
            row["temp_7d_mean"] = float(arrays["temp_avg_c"][window_7].mean())
            row["temp_14d_mean"] = float(arrays["temp_avg_c"][window_14].mean())
            row["temp_21d_mean"] = float(arrays["temp_avg_c"][window_21].mean())
            row["temp_28d_mean"] = float(arrays["temp_avg_c"][window_28].mean())
            row["temp_range_24h_c"] = float(arrays["temp_range_24h_c"][current_idx])

            row["relative_humidity"] = float(arrays["relative_humidity"][current_idx])
            row["relative_humidity_max"] = float(arrays["relative_humidity_max"][current_idx])
            row["relative_humidity_min"] = float(arrays["relative_humidity_min"][current_idx])

            # row["rh_3d_mean"] = float(arrays["relative_humidity"][window_3].mean())
            row["rh_7d_mean"] = float(arrays["relative_humidity"][window_7].mean())
            row["rh_14d_mean"] = float(arrays["relative_humidity"][window_14].mean())
            row["rh_21d_mean"] = float(arrays["relative_humidity"][window_21].mean())
            row["rh_28d_mean"] = float(arrays["relative_humidity"][window_28].mean())
            row["humidity_range_daily"] = float(arrays["humidity_range_daily"][current_idx])
        
            row["soil_rel_humidity_7d_mean"] = float(arrays["soil_rel_humidity"][window_7].mean())
            row["soil_rel_humidity_14d_mean"] = float(arrays["soil_rel_humidity"][window_14].mean())
            row["soil_rel_humidity_21d_mean"] = float(arrays["soil_rel_humidity"][window_21].mean())
            row["soil_rel_humidity_28d_mean"] = float(arrays["soil_rel_humidity"][window_28].mean())

            row["radiation_7d_mean"] = float(arrays["radiation_avg"][window_7].mean())
            row["radiation_28d_mean"] = float(arrays["radiation_avg"][window_28].mean())
            row["wind_7d_mean"] = float(arrays["wind_avg"][window_7].mean())
            row["wind_28d_mean"] = float(arrays["wind_avg"][window_28].mean())
            row["is_weak_wind_day"] = float(arrays["weak_wind_flag"][current_idx])
            row["weak_wind_streak_days"] = float(arrays["weak_wind_streak_days"][current_idx])
            row["low_radiation_streak_days"] = float(arrays["low_radiation_streak_days"][current_idx])

            row["temp_low_threshold_site"] = float(temp_thresholds["low"])
            row["temp_high_threshold_site"] = float(temp_thresholds["high"])

            row["hot_streak_days"] = float(arrays["temp_high_streak_days"][current_idx])
            row["cold_streak_days"] = float(arrays["temp_low_streak_days"][current_idx])
            row["optimal_temp_streak_days"] = float(arrays["temp_optimal_streak_days"][current_idx])

            row["high_humidity_streak_days"] = float(arrays["high_humidity_streak_days"][current_idx])
            row["medium_high_humidity_streak_days"] = float(arrays["medium_high_humidity_streak_days"][current_idx])

            row["high_humidity_7d_count"] = float(arrays["high_humidity_flag"][window_7].sum())
            row["high_humidity_28d_count"] = float(arrays["high_humidity_flag"][window_28].sum())
            # row["high_humidity_3d_count"] = float(arrays["high_humidity_flag"][window_3].sum())

            # row["heavy_rain_3d_count"] = float(arrays["heavy_rain_flag"][window_3].sum())
            row["heavy_rain_7d_count"] = float(arrays["heavy_rain_flag"][window_7].sum())
            row["heavy_rain_28d_count"] = float(arrays["heavy_rain_flag"][window_28].sum())
            row["heavy_rain_streak_days"] = float(arrays["heavy_rain_streak_days"][current_idx])
            row["max_single_day_rain_7d"] = float(arrays["precip_sum"][window_7].max())
            row["max_single_day_rain_28d"] = float(arrays["precip_sum"][window_28].max())

            row["hot_humid_streak_days"] = float(arrays["hot_humid_streak_days"][current_idx])
            row["optimal_temp_humid_streak_days"] = float(arrays["optimal_temp_humid_streak_days"][current_idx])
            row["weak_wind_humid_streak_days"] = float(arrays["weak_wind_humid_streak_days"][current_idx])

    return grouped

def build_weather_process_feature_rows(weather_by_site: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    raw_only = bool(getattr(cfg, "USE_RAW_ONLY_FEATURES", False))
    rows: list[dict[str, Any]] = []

    for site_id in sorted(weather_by_site):
        weather_site = weather_by_site[site_id]
        arrays = weather_site["arrays"]
        meta = weather_site["meta"]
        records = weather_site["records"]
        temp_thresholds = weather_site["temp_thresholds"]

        for current_idx, record in enumerate(records):
            window_3 = slice(max(0, current_idx - 2), current_idx + 1)
            window_7 = slice(max(0, current_idx - 6), current_idx + 1)
            window_14 = slice(max(0, current_idx - 13), current_idx + 1)
            window_21 = slice(max(0, current_idx - 20), current_idx + 1)
            window_28 = slice(max(0, current_idx - 27), current_idx + 1)
            cumulative = slice(0, current_idx + 1)

            row = {
                "site_id": site_id,
                "site_name": meta["canonical_name"],
                "date_str": record["date"].isoformat(),

                # ===== 原始气象字段 =====
                "wind_avg": float(arrays["wind_avg"][current_idx]),
                "wind_max": float(arrays["wind_max"][current_idx]),
                "wind_min": float(arrays["wind_min"][current_idx]),

                "precip_max": float(arrays["precip_max"][current_idx]),
                "precip_min": float(arrays["precip_min"][current_idx]),
                "precip_sum": float(arrays["precip_sum"][current_idx]),

                # "humidity_avg": float(arrays["humidity_avg"][current_idx]),
                # "humidity_max": float(arrays["humidity_max"][current_idx]),
                # "humidity_min": float(arrays["humidity_min"][current_idx]),

                "relative_humidity": float(arrays["relative_humidity"][current_idx]),
                "relative_humidity_max": float(arrays["relative_humidity_max"][current_idx]),
                "relative_humidity_min": float(arrays["relative_humidity_min"][current_idx]),

                "temp_avg_c": float(arrays["temp_avg_c"][current_idx]),
                "temp_max_c": float(arrays["temp_max_c"][current_idx]),
                "temp_min_c": float(arrays["temp_min_c"][current_idx]),

                "soil_moisture": float(arrays["soil_moisture"][current_idx]),

                "surface_temp_avg_c": float(arrays["surface_temp_avg_c"][current_idx]),
                "surface_temp_max_c": float(arrays["surface_temp_max_c"][current_idx]),
                "surface_temp_min_c": float(arrays["surface_temp_min_c"][current_idx]),

                "pressure_kpa": float(arrays["pressure_kpa"][current_idx]),
                "pressure_max_kpa": float(arrays["pressure_max_kpa"][current_idx]),
                "pressure_min_kpa": float(arrays["pressure_min_kpa"][current_idx]),

                "radiation_avg": float(arrays["radiation_avg"][current_idx]),
                "radiation_max": float(arrays["radiation_max"][current_idx]),
                "radiation_min": float(arrays["radiation_min"][current_idx]),

                "soil_rel_humidity": float(arrays["soil_rel_humidity"][current_idx]),
                "soil_temp_c": float(arrays["soil_temp_c"][current_idx]),
            }

            if not raw_only:
                row.update(
                    {
                        # ===== 过程特征 =====
                        "gdd_cum": float(np.maximum(arrays["temp_avg_c"][cumulative] - 10.0, 0.0).sum()),
                        "rain_7d_sum": float(arrays["precip_sum"][window_7].sum()),
                        "rain_14d_sum": float(arrays["precip_sum"][window_14].sum()),
                        "rain_21d_sum": float(arrays["precip_sum"][window_21].sum()),
                        "rain_28d_sum": float(arrays["precip_sum"][window_28].sum()),
                        "rainy_streak_days": float(arrays["rainy_streak_days"][current_idx]),
                        "rain_gap_days": float(arrays["rain_gap_days"][current_idx]),
                        "temp_7d_mean": float(arrays["temp_avg_c"][window_7].mean()),
                        "temp_14d_mean": float(arrays["temp_avg_c"][window_14].mean()),
                        "temp_21d_mean": float(arrays["temp_avg_c"][window_21].mean()),
                        "temp_28d_mean": float(arrays["temp_avg_c"][window_28].mean()),
                        "temp_range_24h_c": float(arrays["temp_range_24h_c"][current_idx]),
                        "rh_7d_mean": float(arrays["relative_humidity"][window_7].mean()),
                        "rh_14d_mean": float(arrays["relative_humidity"][window_14].mean()),
                        "rh_21d_mean": float(arrays["relative_humidity"][window_21].mean()),
                        "rh_28d_mean": float(arrays["relative_humidity"][window_28].mean()),
                        "humidity_range_daily": float(arrays["humidity_range_daily"][current_idx]),
                        "soil_rel_humidity_14d_mean": float(arrays["soil_rel_humidity"][window_14].mean()),
                        "soil_rel_humidity_7d_mean": float(arrays["soil_rel_humidity"][window_7].mean()),
                        "soil_rel_humidity_21d_mean": float(arrays["soil_rel_humidity"][window_21].mean()),
                        "soil_rel_humidity_28d_mean": float(arrays["soil_rel_humidity"][window_28].mean()),
                        "radiation_7d_mean": float(arrays["radiation_avg"][window_7].mean()),
                        "radiation_28d_mean": float(arrays["radiation_avg"][window_28].mean()),
                        "wind_7d_mean": float(arrays["wind_avg"][window_7].mean()),
                        "wind_28d_mean": float(arrays["wind_avg"][window_28].mean()),
                        "is_weak_wind_day": float(arrays["weak_wind_flag"][current_idx]),
                        "weak_wind_streak_days": float(arrays["weak_wind_streak_days"][current_idx]),
                        "low_radiation_streak_days": float(arrays["low_radiation_streak_days"][current_idx]),
                        "temp_low_threshold_site": float(temp_thresholds["low"]),
                        "temp_high_threshold_site": float(temp_thresholds["high"]),
                        "hot_streak_days": float(arrays["temp_high_streak_days"][current_idx]),
                        "cold_streak_days": float(arrays["temp_low_streak_days"][current_idx]),
                        "optimal_temp_streak_days": float(arrays["temp_optimal_streak_days"][current_idx]),
                        "high_humidity_streak_days": float(arrays["high_humidity_streak_days"][current_idx]),
                        "medium_high_humidity_streak_days": float(arrays["medium_high_humidity_streak_days"][current_idx]),
                        "high_humidity_7d_count": float(arrays["high_humidity_flag"][window_7].sum()),
                        "high_humidity_28d_count": float(arrays["high_humidity_flag"][window_28].sum()),
                        "heavy_rain_7d_count": float(arrays["heavy_rain_flag"][window_7].sum()),
                        "heavy_rain_28d_count": float(arrays["heavy_rain_flag"][window_28].sum()),
                        "heavy_rain_streak_days": float(arrays["heavy_rain_streak_days"][current_idx]),
                        "max_single_day_rain_7d": float(arrays["precip_sum"][window_7].max()),
                        "max_single_day_rain_28d": float(arrays["precip_sum"][window_28].max()),
                        "hot_humid_streak_days": float(arrays["hot_humid_streak_days"][current_idx]),
                        "optimal_temp_humid_streak_days": float(arrays["optimal_temp_humid_streak_days"][current_idx]),
                        "weak_wind_humid_streak_days": float(arrays["weak_wind_humid_streak_days"][current_idx]),
                    }
                )

            rows.append(row)

    return rows

def split_rows_by_replicate(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[row.get("replicate_id_same_day", 1)].append(row)

    sub_sequences: list[list[dict[str, Any]]] = []
    for replicate_id in sorted(grouped):
        subset = grouped[replicate_id]
        subset.sort(key=lambda item: (item["date"], item.get("record_id", 0)))
        sub_sequences.append(subset)

    return sub_sequences


def build_training_arrays(
    site_rows: dict[int, list[dict[str, Any]]], site_ids: list[int], targets: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seqs = []
    tabs = []
    ys = []
    masks = []
    prev_targets_list = []

    for site_id in site_ids:
        rows = site_rows[site_id]
        sub_sequences = split_rows_by_replicate(rows)

        for subset in sub_sequences:
            if len(subset) < 2:
                continue

            for idx in range(1, len(subset)):
                current = subset[idx]
                previous = subset[idx - 1]

                prev_targets = np.asarray(
                    [fill_none(previous[targets[0]]) / 100.0, fill_none(previous[targets[1]]) / 100.0],
                    dtype=np.float32,
                )
                current_targets = np.asarray(
                    [fill_none(current[targets[0]]) / 100.0, fill_none(current[targets[1]]) / 100.0],
                    dtype=np.float32,
                )

                seqs.append(current["weather_seq_28"])

                tab_values = [fill_none(current[name]) for name in cfg.BASE_MODEL_FEATURES]
                tab_values.extend(prev_targets.tolist())
                tabs.append(tab_values)

                ys.append(current_targets - prev_targets)
                prev_targets_list.append(prev_targets)

                masks.append(
                    [
                        1.0 if current[targets[0]] is not None else 0.0,
                        1.0 if current[targets[1]] is not None else 0.0,
                    ]
                )

    return (
        np.asarray(seqs, dtype=np.float32),
        np.asarray(tabs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(masks, dtype=np.float32),
        np.asarray(prev_targets_list, dtype=np.float32),
    )
# [03.5] ===== 训练数组构建 =====
# def build_training_arrays(
#     site_rows: dict[int, list[dict[str, Any]]], site_ids: list[int], targets: list[str]
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     seqs = []
#     tabs = []
#     ys = []
#     masks = []
#     prev_targets_list = []
#     for site_id in site_ids:
#         rows = site_rows[site_id]
#         for idx in range(1, len(rows)):
#             current = rows[idx]
#             previous = rows[idx - 1]
#             prev_targets = np.asarray(
#                 [fill_none(previous[targets[0]]) / 100.0, fill_none(previous[targets[1]]) / 100.0],
#                 dtype=np.float32,
#             )
#             current_targets = np.asarray(
#                 [fill_none(current[targets[0]]) / 100.0, fill_none(current[targets[1]]) / 100.0],
#                 dtype=np.float32,
#             )
#             seqs.append(current["weather_seq_21"])
#             tab_values = [fill_none(current[name]) for name in cfg.BASE_MODEL_FEATURES]
#             tab_values.extend(prev_targets.tolist())
#             tabs.append(tab_values)
#             ys.append(current_targets - prev_targets)
#             prev_targets_list.append(prev_targets)
#             masks.append(
#                 [
#                     1.0 if current[targets[0]] is not None else 0.0,
#                     1.0 if current[targets[1]] is not None else 0.0,
#                 ]
#             )
#     return (
#         np.asarray(seqs, dtype=np.float32),
#         np.asarray(tabs, dtype=np.float32),
#         np.asarray(ys, dtype=np.float32),
#         np.asarray(masks, dtype=np.float32),
#         np.asarray(prev_targets_list, dtype=np.float32),
#     )


# [03.6] ===== 标准化 =====
def fit_scalers(train_seq: np.ndarray, train_tab: np.ndarray, train_y: np.ndarray) -> dict[str, np.ndarray]:
    seq_mean = train_seq.reshape(-1, train_seq.shape[-1]).mean(axis=0)
    seq_std = train_seq.reshape(-1, train_seq.shape[-1]).std(axis=0)
    tab_mean = train_tab.mean(axis=0)
    tab_std = train_tab.std(axis=0)
    target_mean = train_y.mean(axis=0)
    target_std = train_y.std(axis=0)
    seq_std = np.where(seq_std < 1e-6, 1.0, seq_std)
    tab_std = np.where(tab_std < 1e-6, 1.0, tab_std)
    target_std = np.where(target_std < 1e-6, 1.0, target_std)
    return {
        "seq_mean": seq_mean.astype(np.float32),
        "seq_std": seq_std.astype(np.float32),
        "tab_mean": tab_mean.astype(np.float32),
        "tab_std": tab_std.astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "target_std": target_std.astype(np.float32),
    }


def apply_scalers(seq: np.ndarray, tab: np.ndarray, scalers: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    scaled_seq = (seq - scalers["seq_mean"]) / scalers["seq_std"]
    scaled_tab = (tab - scalers["tab_mean"]) / scalers["tab_std"]
    return scaled_seq.astype(np.float32), scaled_tab.astype(np.float32)


def scale_targets(targets: np.ndarray, scalers: dict[str, np.ndarray]) -> np.ndarray:
    scaled = (targets - scalers["target_mean"]) / scalers["target_std"]
    return scaled.astype(np.float32)


def unscale_targets(targets: np.ndarray, scalers: dict[str, np.ndarray]) -> np.ndarray:
    return targets * scalers["target_std"] + scalers["target_mean"]
