from __future__ import annotations

# [06.1] ===== 动态加载分模块（文件名前缀 01~05） =====
from pathlib import Path
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
ar = _load_module("05_analysis_and_reporting.py", "ar")


# [06.1.1] ===== 导出字段中文化 =====
def to_cn_rows(rows: list[dict], mapping: dict[str, str]) -> list[dict]:
    converted = []
    for row in rows:
        converted.append({mapping.get(k, k): v for k, v in row.items()})
    return converted


def localize_feature_values(rows: list[dict]) -> list[dict]:
    """将行内特征名字段值转中文（仅替换值，不改字段名）。"""
    localized = []
    for row in rows:
        new_row = dict(row)
        if "feature" in new_row and isinstance(new_row["feature"], str):
            # new_row["feature"] = new_row["feature"]  # 原逻辑（保留，不删除）
            new_row["feature"] = cfg.FEATURE_LABELS.get(new_row["feature"], new_row["feature"])
        localized.append(new_row)
    return localized


CN_COLS = {
    "mode": "模式",
    "site_id": "点位编号",
    "site_alias": "点位别名",
    "site_name": "点位名称",
    "date_str": "日期",
    "survey_order": "调查序号",
    "is_baseline": "是否基线点",
    "target": "目标变量",
    "target_cn": "目标变量中文",
    "n": "样本数",
    "mae": "平均绝对误差",
    "rmse": "均方根误差",
    "r2": "决定系数R2",
    "feature": "特征变量",
    "feature_cn": "特征变量中文",
    "n_pairs": "配对样本数",
    "spearman_rho": "Spearman系数",
    "spearman_p": "Spearman显著性p值",
    "pearson_r": "Pearson系数",
    "pearson_p": "Pearson显著性p值",
    "rank": "排名",
    "disease": "病害类型",
    "disease_cn": "病害中文",
    "baseline_mean_rmse": "基线均值RMSE",
    "permuted_mean_rmse": "置换后均值RMSE",
    "importance_delta_rmse": "重要性增量RMSE",
    "baseline_output1_rmse": "基线输出1_RMSE",
    "baseline_output2_rmse": "基线输出2_RMSE",
    "permuted_output1_rmse": "置换输出1_RMSE",
    "permuted_output2_rmse": "置换输出2_RMSE",
    "raw_date": "原始日期",
    "adjusted_date": "修正日期",
    "shift_days": "日期偏移天数",
    "raw_site_name": "原始点位名",
    "variety": "品种",
    "sample_count": "重复数",
    "varieties": "品种集合",
    "stage_label": "生育期标签",
    "stage_code": "生育期编码",
    "gray_incidence": "灰斑病发病株率",
    "gray_index": "灰斑病病情指数",
    "blight_incidence": "大斑病发病株率",
    "blight_index": "大斑病病情指数",
    "white_incidence": "白斑病发病株率",
    "white_index": "白斑病病情指数",
    "actual_gray_incidence": "实际_灰斑病发病株率",
    "actual_gray_index": "实际_灰斑病病情指数",
    "actual_blight_incidence": "实际_大斑病发病株率",
    "actual_blight_index": "实际_大斑病病情指数",
    "actual_white_incidence": "实际_白斑病发病株率",
    "actual_white_index": "实际_白斑病病情指数",
    "pred_gray_incidence": "预测_灰斑病发病株率",
    "pred_gray_index": "预测_灰斑病病情指数",
    "pred_blight_incidence": "预测_大斑病发病株率",
    "pred_blight_index": "预测_大斑病病情指数",
    "pred_white_incidence": "预测_白斑病发病株率",
    "pred_white_index": "预测_白斑病病情指数",
    "shift_days_total": "累计偏移天数",
    "shifted_row_count": "偏移记录数",
    "days_since_first": "距首次调查天数",
    "days_since_prev": "距上次调查天数",

    "record_id": "记录编号",
    "replicate_id_same_day": "同日重复组编号",

    "wind_avg": "原始_10m风速均值",
    "wind_max": "原始_10m风速最大值",
    "wind_min": "原始_10m风速最小值",

    "precip_max": "原始_24小时最大降水量",
    "precip_min": "原始_24小时最小降水量",
    "precip_sum": "原始_24小时降水量之和",

    "relative_humidity": "原始_相对湿度均值",
    "relative_humidity_max": "原始_相对湿度最大值",
    "relative_humidity_min": "原始_相对湿度最小值",

    "temp_avg_c": "原始_2m气温均值(℃)",
    "temp_max_c": "原始_2m气温最大值(℃)",
    "temp_min_c": "原始_2m气温最小值(℃)",

    "soil_moisture": "原始_土壤湿度",

    "surface_temp_avg_c": "原始_地表温度均值(℃)",
    "surface_temp_max_c": "原始_地表温度最大值(℃)",
    "surface_temp_min_c": "原始_地表温度最小值(℃)",

    "pressure_kpa": "原始_地面气压均值(kPa)",
    "pressure_max_kpa": "原始_地面气压最大值(kPa)",
    "pressure_min_kpa": "原始_地面气压最小值(kPa)",

    "radiation_avg": "原始_短波辐射均值",
    "radiation_max": "原始_短波辐射最大值",
    "radiation_min": "原始_短波辐射最小值",

    "soil_rel_humidity": "原始_0-10cm土壤相对湿度",
    "soil_temp_c": "原始_5cm土壤温度(℃)",

    "gdd_cum": "有效积温_GDD",
    "rain_3d_sum": "累计降水_3d(mm)",
    "rain_7d_sum": "累计降水_7d(mm)",
    "rain_14d_sum": "累计降水_14d(mm)",
    "rainy_streak_days": "连续降雨天数",
    "rain_gap_days": "降雨间歇长度(天)",

    # "temp_3d_mean": "平均气温℃_3d",
    "temp_7d_mean": "平均气温℃_7d",
    "temp_14d_mean": "平均气温℃_14d",
    "temp_21d_mean": "平均气温℃_21d",
    "temp_range_24h_c": "24h温差℃",

    # "rh_3d_mean": "平均相对湿度_3d",
    "rh_7d_mean": "平均相对湿度_7d",
    "rh_14d_mean": "平均相对湿度_14d",
    "rh_21d_mean": "平均相对湿度_21d",
    "humidity_range_daily": "湿度日较差",
    
    "soil_rel_humidity_14d_mean": "平均土壤相对湿度_14d",
    "soil_rel_humidity_7d_mean": "平均土壤相对湿度_7d",
    "soil_rel_humidity_21d_mean": "平均土壤相对湿度_21d",

    "radiation_7d_mean": "平均短波辐射_7d",
    "wind_7d_mean": "平均风速_7d",

    "is_weak_wind_day": "是否弱风日",
    "weak_wind_streak_days": "弱风日连续天数",
    "low_radiation_streak_days": "寡照连续天数",
    

    "temp_low_threshold_site": "站点低温阈值(℃)",
    "temp_high_threshold_site": "站点高温阈值(℃)",
    "hot_streak_days": "连续高温天数",
    "cold_streak_days": "连续低温天数",
    "optimal_temp_streak_days": "连续适温天数",

    "high_humidity_streak_days": "连续高湿天数",
    "medium_high_humidity_streak_days": "连续较高湿度天数",
    "high_humidity_7d_count": "7天内高湿天数",
    # "high_humidity_3d_count": "3天内高湿天数",

    # "heavy_rain_3d_count": "3天内强降雨次数",
    "heavy_rain_7d_count": "7天内强降雨次数",
    "heavy_rain_streak_days": "连续强降雨次数",
    "max_single_day_rain_7d": "7天最大单日降雨_mm",

    "hot_humid_streak_days": "连续高温高湿天数",
    "optimal_temp_humid_streak_days": "连续适温高湿天数",
    "weak_wind_humid_streak_days": "连续弱风高湿天数",

    "actual_gray_incidence_risk": "实际_灰斑病发病株率风险",
    "pred_gray_incidence_risk": "预测_灰斑病发病株率风险",
    "actual_gray_index_risk": "实际_灰斑病病情指数风险",
    "pred_gray_index_risk": "预测_灰斑病病情指数风险",

    "actual_blight_incidence_risk": "实际_大斑病发病株率风险",
    "pred_blight_incidence_risk": "预测_大斑病发病株率风险",
    "actual_blight_index_risk": "实际_大斑病病情指数风险",
    "pred_blight_index_risk": "预测_大斑病病情指数风险",

    "actual_white_incidence_risk": "实际_白斑病发病株率风险",
    "pred_white_incidence_risk": "预测_白斑病发病株率风险",
    "actual_white_index_risk": "实际_白斑病病情指数风险",
    "pred_white_index_risk": "预测_白斑病病情指数风险",

    "actual_overall_risk": "实际_综合风险",
    "pred_overall_risk": "预测_综合风险",
    }


# [06.2] ===== 主流程 =====
def main() -> None:
    # [06.2.1] 初始化
    cfg.set_global_seed(cfg.RANDOM_SEED)
    cfg.ensure_dirs()

    model_dir = cfg.MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    # [06.2.2] 读取原始输入
    survey_path, weather_path = dc.unzip_inputs()
    docx_text = dc.read_docx_text(cfg.DOCX_PATH)
    (cfg.OUT_DIR / "docx_extracted_text.txt").write_text(docx_text, encoding="utf-8")

    # [06.2.3] 数据清洗与聚合
    metadata, _ = dc.read_station_metadata(weather_path)
    weather_by_site = dc.read_weather_series(weather_path, metadata)
    panel_rows, quality_rows = dc.read_and_aggregate_survey(survey_path, weather_by_site)

    # # [06.2.4] 特征工程
    # site_rows = fe.add_process_features(panel_rows, weather_by_site)
    # flat_panel_rows = [row for site_id in sorted(site_rows) for row in site_rows[site_id]]
    
    # [06.2.4] 特征工程
    site_rows = fe.add_process_features(panel_rows, weather_by_site)
    flat_panel_rows = [row for site_id in sorted(site_rows) for row in site_rows[site_id]]
    # ===== 新增：基于气象原始表生成“原始气象 + 过程特征”总表 =====
    weather_process_rows = fe.build_weather_process_feature_rows(weather_by_site)
    


    # [06.2.5] 分病害训练 + 留一站点交叉验证 + 全量拟合
    cv_prediction_sets = []
    full_prediction_sets = []
    importance_by_disease: dict[str, list[dict]] = {}

    for disease_key, disease_conf in cfg.DISEASE_CONFIGS.items():
        targets = disease_conf["targets"]
        cv_rows = []
        bundles_by_site: dict[int, dict] = {}

        for held_out_site in sorted(site_rows.keys()):
            remaining_sites = [site_id for site_id in sorted(site_rows.keys()) if site_id != held_out_site]
            train_sites, validation_sites = mt.split_train_validation_sites(remaining_sites, cfg.RANDOM_SEED + held_out_site)
            bundle = mt.train_model(site_rows, train_sites, validation_sites, targets, cfg.RANDOM_SEED + held_out_site)
            bundles_by_site[held_out_site] = bundle
            held_out_predictions = mt.rolling_predictions(
                {held_out_site: site_rows[held_out_site]}, bundle, targets, "cv_optimized", use_actual_previous=True
            )
            cv_rows.extend(held_out_predictions)

        cv_prediction_sets.append(cv_rows)

        importance_rows = ar.compute_feature_importance(site_rows, bundles_by_site, disease_key, targets)
        importance_by_disease[disease_key] = importance_rows
        ar.plot_feature_importance(importance_rows, disease_key)

        full_bundle = mt.train_full_model(site_rows, targets, cfg.RANDOM_SEED + 100 + len(full_prediction_sets))
        model_save_path = cfg.MODEL_DIR / f"fus_full_bundle_{disease_key}.pt"
        mt.save_bundle(full_bundle, model_save_path)    
        full_rows = mt.rolling_predictions(site_rows, full_bundle, targets, "full_fit", use_actual_previous=True)
        full_prediction_sets.append(full_rows)

    # [06.2.6] 融合预测与评估
    merged_predictions = mt.merge_prediction_tables(cv_prediction_sets + full_prediction_sets)
    cv_metrics = mt.compute_metrics(merged_predictions, "cv_optimized")
    corr_rows, top_corr_rows = ar.compute_correlations(flat_panel_rows)
    importance_all_rows = [row for disease_key in cfg.DISEASE_CONFIGS for row in importance_by_disease[disease_key]]

    # [06.2.6.1] 表格行值中文化（特征名）
    # corr_rows_cn = corr_rows  # 原逻辑（保留，不删除）
    # top_corr_rows_cn = top_corr_rows  # 原逻辑（保留，不删除）
    # importance_all_rows_cn = importance_all_rows  # 原逻辑（保留，不删除）
    corr_rows_cn = localize_feature_values(corr_rows)
    top_corr_rows_cn = localize_feature_values(top_corr_rows)
    importance_all_rows_cn = localize_feature_values(importance_all_rows)
    importance_by_disease_cn = {k: localize_feature_values(v) for k, v in importance_by_disease.items()}

    # [06.2.7] 绘图
    ar.plot_curves(merged_predictions, "full_fit")
    ar.plot_cv_scatter(merged_predictions)
    ar.plot_correlation_heatmap(corr_rows_cn)

    # [06.2.8] 导出结果
    clean_panel_export = []
    # excluded_export_keys = {"weather_seq_21", "date", "gray_resistance", "blight_resistance", "white_resistance"}
    excluded_export_keys = {
    "weather_seq_28",
    "date",
    "gray_resistance",
    "blight_resistance",
    "white_resistance",
    "sample_count",
    "survey_order",
    "days_since_first",
    "days_since_prev",
    }
    for row in flat_panel_rows:
        export_row = {k: v for k, v in row.items() if k not in excluded_export_keys}
        clean_panel_export.append(export_row)
    
    # ===== 新增：导出“全部特征总表” =====
    all_feature_export = []
    excluded_all_feature_keys = {
        "weather_seq_28",
        "date",
        "gray_resistance",
        "blight_resistance",
        "white_resistance",

        # 不展示2m比湿中间字段
        "humidity_avg",
        "humidity_max",
        "humidity_min",
    }

    for row in flat_panel_rows:
        export_row = {k: v for k, v in row.items() if k not in excluded_all_feature_keys}
        all_feature_export.append(export_row)


    dc.write_csv(cfg.OUT_DIR / "气象原始_过程特征表.csv", to_cn_rows(weather_process_rows, CN_COLS))
    dc.write_csv(cfg.OUT_DIR / "全部特征总表.csv", to_cn_rows(all_feature_export, CN_COLS))
    dc.write_csv(cfg.OUT_DIR / "清洗后面板数据.csv", to_cn_rows(clean_panel_export, CN_COLS))
    dc.write_csv(cfg.OUT_DIR / "质量检查日志.csv", to_cn_rows(quality_rows, CN_COLS))
    dc.write_csv(
        cfg.OUT_DIR / "交叉验证预测结果.csv",
        to_cn_rows([row for row in merged_predictions if row["mode"] == "cv_optimized"], CN_COLS),
    )
    dc.write_csv(
        cfg.OUT_DIR / "全量拟合预测结果.csv",
        to_cn_rows([row for row in merged_predictions if row["mode"] == "full_fit"], CN_COLS),
    )
    dc.write_csv(cfg.OUT_DIR / "LSTM交叉验证指标.csv", to_cn_rows(cv_metrics, CN_COLS))
    dc.write_csv(cfg.OUT_DIR / "相关性总表_Spearman.csv", to_cn_rows(corr_rows_cn, CN_COLS))
    dc.write_csv(cfg.OUT_DIR / "相关性Top10.csv", to_cn_rows(top_corr_rows_cn, CN_COLS))
    dc.write_csv(cfg.OUT_DIR / "特征重要性总表.csv", to_cn_rows(importance_all_rows_cn, CN_COLS))
    for disease_key in cfg.DISEASE_CONFIGS:
        # dc.write_csv(cfg.OUT_DIR / f"feature_importance_{disease_key}.csv", importance_by_disease[disease_key])
        disease_cn = cfg.DISEASE_CONFIGS[disease_key]["cn_name"]
        dc.write_csv(cfg.OUT_DIR / f"特征重要性_{disease_cn}.csv", to_cn_rows(importance_by_disease_cn[disease_key], CN_COLS))

    # dc.write_workbook(
    #     cfg.OUT_DIR / "analysis_summary.xlsx",
    dc.write_workbook(
        cfg.OUT_DIR / "分析结果汇总.xlsx",
        [
            ("气象原始_过程特征表", to_cn_rows(weather_process_rows, CN_COLS)),
            ("全部特征总表", to_cn_rows(all_feature_export, CN_COLS)),
            ("清洗后面板", to_cn_rows(clean_panel_export, CN_COLS)),
            ("质量检查日志", to_cn_rows(quality_rows, CN_COLS)),
            ("交叉验证预测", to_cn_rows([row for row in merged_predictions if row["mode"] == "cv_optimized"], CN_COLS)),
            ("全量拟合预测", to_cn_rows([row for row in merged_predictions if row["mode"] == "full_fit"], CN_COLS)),
            ("交叉验证指标", to_cn_rows(cv_metrics, CN_COLS)),
            ("相关性总表", to_cn_rows(corr_rows_cn, CN_COLS)),
            ("相关性Top10", to_cn_rows(top_corr_rows_cn, CN_COLS)),
            ("重要性_灰斑病", to_cn_rows(importance_by_disease_cn["gray"], CN_COLS)),
            ("重要性_大斑病", to_cn_rows(importance_by_disease_cn["blight"], CN_COLS)),
            ("重要性_白斑病", to_cn_rows(importance_by_disease_cn["white"], CN_COLS)),
        ],
    )

    # [06.2.9] 生成报告
    report_text = ar.build_report(docx_text, flat_panel_rows, quality_rows, cv_metrics, top_corr_rows_cn, importance_all_rows_cn)
    (cfg.OUT_DIR / "README_结果说明.md").write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
