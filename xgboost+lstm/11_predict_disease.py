from __future__ import annotations

"""
基于已训练好的 fus_full_bundle_*.pt 执行病害预测。

默认行为：
1) 读取 data 下调查/气象数据
2) 复用现有清洗 + 特征工程逻辑
3) 加载 models/fus_full_bundle_gray.pt / blight / white
4) 输出融合后的预测表到 results_leafspot_lstm/预测结果_推理.csv
"""

from pathlib import Path
import argparse
import importlib.util


def _load_module(file_name: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


cfg = _load_module("01_config.py", "cfg")
dc = _load_module("02_data_cleaning.py", "dc")
fe = _load_module("03_feature_engineering.py", "fe")
mt = _load_module("04_model_training_testing.py", "mt")


def run_prediction(use_actual_previous: bool, mode_name: str, output_path: Path) -> None:
    cfg.set_global_seed(cfg.RANDOM_SEED)
    cfg.ensure_dirs()

    survey_path, weather_path = dc.unzip_inputs()
    metadata, _ = dc.read_station_metadata(weather_path)
    weather_by_site = dc.read_weather_series(weather_path, metadata)
    panel_rows, _ = dc.read_and_aggregate_survey(survey_path, weather_by_site)
    site_rows = fe.add_process_features(panel_rows, weather_by_site)

    prediction_sets: list[list[dict]] = []
    for disease_key, disease_conf in cfg.DISEASE_CONFIGS.items():
        bundle_path = cfg.MODEL_DIR / f"fus_full_bundle_{disease_key}.pt"
        if not bundle_path.exists():
            raise FileNotFoundError(f"未找到模型文件: {bundle_path}")

        bundle = mt.load_bundle(bundle_path)
        targets = disease_conf["targets"]

        rows = mt.rolling_predictions(
            site_rows=site_rows,
            bundle=bundle,
            targets=targets,
            mode_name=mode_name,
            use_actual_previous=use_actual_previous,
        )
        prediction_sets.append(rows)

    merged_rows = mt.merge_prediction_tables(prediction_sets)
    dc.write_csv(output_path, merged_rows)

    print(f"预测完成，输出行数: {len(merged_rows)}")
    print(f"输出文件: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="加载已训练模型并执行病害预测")
    parser.add_argument(
        "--use-actual-previous",
        action="store_true",
        help="使用上一期真实值作为下一期输入（评估模式）；默认关闭为纯递推预测模式",
    )
    parser.add_argument(
        "--mode-name",
        type=str,
        default="inference",
        help="输出结果中的 mode 字段值，默认 inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(cfg.OUT_DIR / "预测结果_推理.csv"),
        help="预测结果输出 CSV 路径",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_prediction(
        use_actual_previous=bool(args.use_actual_previous),
        mode_name=args.mode_name,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
