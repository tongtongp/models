from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import defaultdict
import importlib.util

_BASE_DIR = Path(__file__).resolve().parent
_DATA_DIR = _BASE_DIR / "data"
_OUTPUT_FILE = _DATA_DIR / "corn_disease.csv"

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


def _build_export_rows(site_rows: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_features = list(cfg.BASE_MODEL_FEATURES)
    target_fields = [
        "gray_incidence",
        "gray_index",
        "blight_incidence",
        "blight_index",
        "white_incidence",
        "white_index",
    ]

    for site_id in sorted(site_rows):
        series_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for row in site_rows[site_id]:
            site_name = str(row.get("site_name") or f"site_{site_id}").strip()
            varieties = str(row.get("varieties") or "").strip()
            if not varieties:
                varieties = f"replicate_{row.get('replicate_id_same_day', 1)}"

            disease_series_id = f"{site_id}|{site_name}|{varieties}"
            series_groups[disease_series_id].append(row)

        for disease_series_id, series_rows in sorted(series_groups.items()):
            series_rows = sorted(
                series_rows,
                key=lambda r: (
                    r.get("date"),
                    r.get("record_id", 0)
                )
            )

            for index in range(1, len(series_rows)):
                row = series_rows[index]
                previous = series_rows[index - 1]

                export_row = {
                    "site_id": row.get("site_id"),
                    "site_name": row.get("site_name"),
                    "varieties": row.get("varieties"),
                    "disease_series_id": disease_series_id,
                    "record_id": row.get("record_id"),
                    "replicate_id_same_day": row.get("replicate_id_same_day"),
                    "date": row.get("date"),
                    "date_str": row.get("date_str"),
                    "weather_seq_28": row.get("weather_seq_28"),
                }

                for field in base_features:
                    export_row[field] = row.get(field)

                for field in target_fields:
                    export_row[field] = row.get(field)
                    prev_value = previous.get(field)
                    export_row[f"prev_{field}"] = prev_value
                    export_row[f"delta_{field}"] = (
                        None
                        if row.get(field) is None or prev_value is None
                        else float(row.get(field)) - float(prev_value)
                    )

                rows.append(export_row)

    return rows


def main() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    survey_path, weather_path = dc.unzip_inputs()
    metadata, _ = dc.read_station_metadata(weather_path)
    weather_by_site = dc.read_weather_series(weather_path, metadata)
    panel_rows, _ = dc.read_and_aggregate_survey(survey_path, weather_by_site)
    site_rows = fe.add_process_features(panel_rows, weather_by_site)

    def rebuild_replicate_groups(site_rows):
    

        new_site_rows = {}

        for site_id, rows in site_rows.items():
            rows = list(rows)

            # 先按日期和原始行号排序，保证同一天第几条记录稳定
            rows.sort(
                key=lambda r: (
                    r["date"],
                    r.get("record_id", 0)
                )
            )

            # 重新生成 replicate_id_same_day
            same_day_counter = defaultdict(int)

            for row in rows:
                key = (row["site_id"], row["date"])
                same_day_counter[key] += 1
                row["replicate_id_same_day"] = same_day_counter[key]

            # 最终按 site_id + replicate_id_same_day + date 排序
            rows.sort(
                key=lambda r: (
                    r["site_id"],
                    r.get("replicate_id_same_day", 1),
                    r["date"],
                    r.get("record_id", 0)
                )
            )

            new_site_rows[site_id] = rows

        return new_site_rows
    export_rows = _build_export_rows(site_rows)
    if not export_rows:
        raise ValueError("未生成任何样本，请检查输入数据与字段完整性。")

    field_order = [
        "site_id",
        "site_name",
        "varieties",
        "disease_series_id",
        "record_id",
        "replicate_id_same_day",
        "date",
        "date_str",
        "weather_seq_28",
        *cfg.BASE_MODEL_FEATURES,
        "gray_incidence",
        "gray_index",
        "blight_incidence",
        "blight_index",
        "white_incidence",
        "white_index",
        "prev_gray_incidence",
        "prev_gray_index",
        "prev_blight_incidence",
        "prev_blight_index",
        "prev_white_incidence",
        "prev_white_index",
        "delta_gray_incidence",
        "delta_gray_index",
        "delta_blight_incidence",
        "delta_blight_index",
        "delta_white_incidence",
        "delta_white_index",
    ]

    dc.write_csv(_OUTPUT_FILE, export_rows, field_order=field_order)
    print(f"已生成：{_OUTPUT_FILE}")
    print(f"样本数：{len(export_rows)}")


if __name__ == "__main__":
    main()

