from __future__ import annotations

# [05.1] ===== 基础库导入 =====
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import importlib.util

# [05.2] ===== 动态加载模块 =====
_cfg_spec = importlib.util.spec_from_file_location("cfg", Path(__file__).with_name("01_config.py"))
cfg = importlib.util.module_from_spec(_cfg_spec)
assert _cfg_spec and _cfg_spec.loader
_cfg_spec.loader.exec_module(cfg)

_fe_spec = importlib.util.spec_from_file_location("fe", Path(__file__).with_name("03_feature_engineering.py"))
fe = importlib.util.module_from_spec(_fe_spec)
assert _fe_spec and _fe_spec.loader
_fe_spec.loader.exec_module(fe)

_mt_spec = importlib.util.spec_from_file_location("mt", Path(__file__).with_name("04_model_training_testing.py"))
mt = importlib.util.module_from_spec(_mt_spec)
assert _mt_spec and _mt_spec.loader
_mt_spec.loader.exec_module(mt)


# [05.3] ===== 置换重要性 =====
def evaluate_prediction_rmse(pred_matrix: np.ndarray, actual_matrix: np.ndarray) -> tuple[float, list[float]]:
    rmse_values = []
    for output_index in range(actual_matrix.shape[1]):
        mask = np.isfinite(actual_matrix[:, output_index])
        if not np.any(mask):
            continue
        rmse = float(np.sqrt(np.mean((pred_matrix[mask, output_index] - actual_matrix[mask, output_index]) ** 2)))
        rmse_values.append(rmse)
    score = float(np.mean(rmse_values)) if rmse_values else float("nan")
    return score, rmse_values

def build_importance_records(
    site_rows: dict[int, list[dict[str, Any]]], bundles_by_site: dict[int, dict[str, Any]], targets: list[str]
) -> list[dict[str, Any]]:
    records = []
    for site_id in sorted(bundles_by_site):
        bundle = bundles_by_site[site_id]
        rows = site_rows[site_id]
        # ===== 关键修改：按 replicate 分组拆成子序列 =====
        sub_sequences = fe.split_rows_by_replicate(rows)
        for subset in sub_sequences:
            if len(subset) < 2:
                continue
            for idx in range(1, len(subset)):
                prev_row = subset[idx - 1]
                current_row = subset[idx]
                prev_targets = np.asarray(
                    [fe.fill_none(prev_row[targets[0]]) / 100.0, fe.fill_none(prev_row[targets[1]]) / 100.0],
                    dtype=np.float32,
                )
                actual_targets = np.asarray(
                    [
                        np.nan if current_row[targets[0]] is None else float(current_row[targets[0]]) / 100.0,
                        np.nan if current_row[targets[1]] is None else float(current_row[targets[1]]) / 100.0,
                    ],
                    dtype=np.float32,
                )
                records.append(
                    {
                        "site_id": site_id,
                        "replicate_id_same_day": current_row.get("replicate_id_same_day", 1),
                        "bundle": bundle,
                        "row": current_row,
                        "prev_targets": prev_targets,
                        "actual_targets": actual_targets,
                    }
                )

    return records
# def build_importance_records(
#     site_rows: dict[int, list[dict[str, Any]]], bundles_by_site: dict[int, dict[str, Any]], targets: list[str]
# ) -> list[dict[str, Any]]:
#     records = []
#     for site_id in sorted(bundles_by_site):
#         bundle = bundles_by_site[site_id]
#         rows = site_rows[site_id]
#         for idx in range(1, len(rows)):
#             prev_row = rows[idx - 1]
#             current_row = rows[idx]
#             prev_targets = np.asarray(
#                 [fe.fill_none(prev_row[targets[0]]) / 100.0, fe.fill_none(prev_row[targets[1]]) / 100.0],
#                 dtype=np.float32,
#             )
#             actual_targets = np.asarray(
#                 [
#                     np.nan if current_row[targets[0]] is None else float(current_row[targets[0]]) / 100.0,
#                     np.nan if current_row[targets[1]] is None else float(current_row[targets[1]]) / 100.0,
#                 ],
#                 dtype=np.float32,
#             )
#             records.append(
#                 {
#                     "site_id": site_id,
#                     "bundle": bundle,
#                     "row": current_row,
#                     "prev_targets": prev_targets,
#                     "actual_targets": actual_targets,
#                 }
#             )
#     return records


def permute_feature_predictions(records: list[dict[str, Any]], feature_name: str, rng: np.random.Generator) -> np.ndarray:
    predictions = []
    if feature_name in cfg.SEQ_FEATURES:
        channel_index = cfg.SEQ_FEATURES.index(feature_name)
        channel_values = [record["row"]["weather_seq_28"][:, channel_index].copy() for record in records]
        permuted_indices = rng.permutation(len(records))
        for record, perm_index in zip(records, permuted_indices):
            row_copy = dict(record["row"])
            row_copy["weather_seq_28"] = record["row"]["weather_seq_28"].copy()
            row_copy["weather_seq_28"][:, channel_index] = channel_values[perm_index]
            predictions.append(mt.predict_row(record["bundle"], row_copy, record["prev_targets"]))
    else:
        values = [fe.fill_none(record["row"].get(feature_name)) for record in records]
        permuted_indices = rng.permutation(len(records))
        for record, perm_index in zip(records, permuted_indices):
            row_copy = dict(record["row"])
            row_copy[feature_name] = float(values[perm_index])
            predictions.append(mt.predict_row(record["bundle"], row_copy, record["prev_targets"]))
    return np.asarray(predictions, dtype=np.float32)


def compute_feature_importance(
    site_rows: dict[int, list[dict[str, Any]]],
    bundles_by_site: dict[int, dict[str, Any]],
    disease_key: str,
    targets: list[str],
) -> list[dict[str, Any]]:
    records = build_importance_records(site_rows, bundles_by_site, targets)
    actual_matrix = np.asarray([record["actual_targets"] for record in records], dtype=np.float32)
    baseline_predictions = np.asarray(
        [mt.predict_row(record["bundle"], record["row"], record["prev_targets"]) for record in records], dtype=np.float32
    )
    baseline_score, baseline_rmse = evaluate_prediction_rmse(baseline_predictions, actual_matrix)

    rows = []
    for feature_index, feature_name in enumerate(cfg.IMPORTANCE_FEATURES):
        rng = np.random.default_rng(cfg.RANDOM_SEED + 1000 + feature_index)
        permuted_predictions = permute_feature_predictions(records, feature_name, rng)
        permuted_score, permuted_rmse = evaluate_prediction_rmse(permuted_predictions, actual_matrix)
        rows.append(
            {
                "disease": disease_key,
                "disease_cn": cfg.DISEASE_CONFIGS[disease_key]["cn_name"],
                "feature": feature_name,
                "feature_cn": cfg.FEATURE_LABELS.get(feature_name, feature_name),
                "baseline_mean_rmse": baseline_score,
                "permuted_mean_rmse": permuted_score,
                "importance_delta_rmse": float(permuted_score - baseline_score),
                "baseline_output1_rmse": baseline_rmse[0] if len(baseline_rmse) > 0 else None,
                "baseline_output2_rmse": baseline_rmse[1] if len(baseline_rmse) > 1 else None,
                "permuted_output1_rmse": permuted_rmse[0] if len(permuted_rmse) > 0 else None,
                "permuted_output2_rmse": permuted_rmse[1] if len(permuted_rmse) > 1 else None,
            }
        )
    rows.sort(key=lambda item: item["importance_delta_rmse"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


# [05.4] ===== 相关性分析 =====
def compute_correlations(panel_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corr_rows = []
    for feature in cfg.CORR_FEATURES:
        for target in cfg.TARGET_LABELS:
            x_values = []
            y_values = []
            for row in panel_rows:
                x = row.get(feature)
                y = row.get(target)
                if x is None or y is None:
                    continue
                x_values.append(float(x))
                y_values.append(float(y))
            if len(x_values) < 4:
                continue
            x_arr = np.asarray(x_values, dtype=np.float64)
            y_arr = np.asarray(y_values, dtype=np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spearman_rho, spearman_p = stats.spearmanr(x_arr, y_arr)
                pearson_r, pearson_p = stats.pearsonr(x_arr, y_arr)
            corr_rows.append(
                {
                    "feature": feature,
                    "feature_cn": cfg.FEATURE_LABELS.get(feature, feature),
                    "target": target,
                    "target_cn": cfg.TARGET_LABELS[target],
                    "n_pairs": int(len(x_arr)),
                    "spearman_rho": None if np.isnan(spearman_rho) else float(spearman_rho),
                    "spearman_p": None if np.isnan(spearman_p) else float(spearman_p),
                    "pearson_r": None if np.isnan(pearson_r) else float(pearson_r),
                    "pearson_p": None if np.isnan(pearson_p) else float(pearson_p),
                }
            )
    top_rows = []
    for target in cfg.TARGET_LABELS:
        subset = [row for row in corr_rows if row["target"] == target and row["spearman_rho"] is not None]
        subset.sort(key=lambda item: (abs(item["spearman_rho"]), item["n_pairs"]), reverse=True)
        for rank, row in enumerate(subset[:10], start=1):
            ranked = dict(row)
            ranked["rank"] = rank
            top_rows.append(ranked)
    return corr_rows, top_rows


# [05.5] ===== 图形输出 =====
def plot_feature_importance(importance_rows: list[dict[str, Any]], disease_key: str) -> None:
    if not importance_rows:
        return
    ordered = sorted(importance_rows, key=lambda item: item["importance_delta_rmse"], reverse=True)[:20]
    # labels = [row["feature"] for row in ordered][::-1]  # 原英文特征名（保留，不删除）
    labels = [row.get("feature_cn", row["feature"]) for row in ordered][::-1]
    values = [row["importance_delta_rmse"] for row in ordered][::-1]
    fig_height = max(6, 0.35 * len(ordered) + 1)
    fig, axis = plt.subplots(figsize=(10, fig_height))
    axis.barh(labels, values, color="#4C78A8")
    # axis.set_xlabel("Permutation importance (delta mean RMSE)")
    # axis.set_ylabel("Feature")
    # axis.set_title(f"{cfg.DISEASE_CONFIGS[disease_key]['prefix'].upper()} feature importance")
    axis.set_xlabel("置换重要性（均值RMSE增量）")
    axis.set_ylabel("特征")
    axis.set_title(f"{cfg.DISEASE_CONFIGS[disease_key]['cn_name']}特征重要性")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    # fig.savefig(cfg.FIG_DIR / f"feature_importance_{disease_key}.png", dpi=220, bbox_inches="tight")
    disease_cn = cfg.DISEASE_CONFIGS[disease_key]["cn_name"]
    fig.savefig(cfg.FIG_DIR / f"特征重要性_{disease_cn}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# def plot_curves(prediction_rows: list[dict[str, Any]], mode_name: str) -> None:
#     target_order = list(cfg.TARGET_LABELS.keys())
#     filtered = [row for row in prediction_rows if row["mode"] == mode_name]
#     for target in target_order:
#         fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharey=False)
#         axes = axes.flatten()
#         for axis in axes:
#             axis.set_visible(False)

#         group_keys = sorted({(row["site_id"], row.get("replicate_id_same_day", 1)) for row in filtered})
#         # for axis_index, site_id in enumerate(sorted({row["site_id"] for row in filtered})):
#         for axis_index, group_key in enumerate(group_keys):
#             site_id, replicate_id = group_key
#             axis = axes[axis_index]
#             axis.set_visible(True)
#             subset = [
#                 row for row in filtered 
#                 if row["site_id"] == site_id and row.get("replicate_id_same_day", 1) == replicate_id
#                 ]
#             # subset.sort(key=lambda item: item["survey_order"])
#             subset.sort(key=lambda item: (item["date_str"], item.get("record_id", 0)))

#             x = np.arange(1, len(subset) + 1)
#             actual = [row.get(f"actual_{target}") for row in subset]
#             pred = [row.get(f"pred_{target}") for row in subset]
#             axis.plot(x, actual, marker="o", linewidth=1.8, label="观测值")
#             axis.plot(x, pred, marker="s", linewidth=1.6, linestyle="--", label="预测值")
#             # axis.set_title(f"点位{site_id} {subset[0]['site_name']}", fontsize=10)
#             axis.set_title(f"{cfg.SITE_ALIAS[site_id]}-第{replicate_id}组", fontsize=10)
#             axis.set_xlabel("调查序号")
#             axis.set_ylabel("数值")
#             axis.grid(alpha=0.25)
#         handles, labels = axes[0].get_legend_handles_labels()
#         fig.legend(handles, labels, loc="upper center", ncol=2)
#         # fig.suptitle(f"{cfg.TARGET_PLOT_LABELS[target]} ({mode_name})", fontsize=16, y=0.98)
#         mode_cn = "全量拟合" if mode_name == "full_fit" else "交叉验证"
#         fig.suptitle(f"{cfg.TARGET_PLOT_LABELS[target]}（{mode_cn}）", fontsize=16, y=0.98)
#         fig.tight_layout(rect=[0, 0, 1, 0.95])
#         # figure_name = f"curve_{mode_name}_{target}.png"
#         figure_name = f"曲线图_{mode_cn}_{cfg.TARGET_LABELS[target]}.png"
#         fig.savefig(cfg.FIG_DIR / figure_name, dpi=220, bbox_inches="tight")
#         plt.close(fig)
def plot_curves(prediction_rows: list[dict[str, Any]], mode_name: str) -> None:
    target_order = list(cfg.TARGET_LABELS.keys())
    filtered = [row for row in prediction_rows if row["mode"] == mode_name]

    if not filtered:
        return

    for target in target_order:
        # ===== 第1步：先按“站点 + 同日调查组”分组 =====
        group_keys = sorted({
            (row["site_id"], row.get("replicate_id_same_day", 1))
            for row in filtered
            if f"actual_{target}" in row or f"pred_{target}" in row
        })

        if not group_keys:
            continue

        # ===== 第2步：根据组数动态创建子图 =====
        n_groups = len(group_keys)
        ncols = 4
        nrows = int(np.ceil(n_groups / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.5 * ncols, 3.2 * nrows),
            sharey=False
        )

        # axes 统一拉平成一维，便于索引
        axes = np.atleast_1d(axes).flatten()

        # 先把所有子图隐藏
        for axis in axes:
            axis.set_visible(False)

        # ===== 第3步：逐组绘图 =====
        for axis_index, group_key in enumerate(group_keys):
            site_id, replicate_id = group_key
            axis = axes[axis_index]
            axis.set_visible(True)

            subset = [
                row for row in filtered
                if row["site_id"] == site_id
                and row.get("replicate_id_same_day", 1) == replicate_id
            ]

            # 按日期 + record_id 排序，保证同日多条记录顺序稳定
            subset.sort(key=lambda item: (item["date_str"], item.get("record_id", 0)))
            
            x = np.arange(1, len(subset) + 1)
            actual = [row.get(f"actual_{target}") for row in subset]
            pred = [row.get(f"pred_{target}") for row in subset]

            axis.plot(x, actual, marker="o", linewidth=1.8, label="观测值")
            axis.plot(x, pred, marker="s", linewidth=1.6, linestyle="--", label="预测值")

            # ===== 纵轴刻度规则：最大值大于30用0-100，否则用0-30 =====
            valid_values = [
                value for value in (actual + pred)
                if value is not None and not np.isnan(value)
            ]
            if valid_values:
                y_max = max(valid_values)
                if y_max > 30:
                    axis.set_ylim(-1, 110)
                else:
                    axis.set_ylim(-1, 30)

            axis.set_title(f"{cfg.SITE_ALIAS[site_id]}-第{replicate_id}组", fontsize=10)
            axis.set_xlabel("调查序号")
            axis.set_ylabel("数值")
            axis.grid(alpha=0.25)

            # x = np.arange(1, len(subset) + 1)
            # actual = [row.get(f"actual_{target}") for row in subset]
            # pred = [row.get(f"pred_{target}") for row in subset]

            # axis.plot(x, actual, marker="o", linewidth=1.8, label="观测值")
            # axis.plot(x, pred, marker="s", linewidth=1.6, linestyle="--", label="预测值")
            # axis.set_title(f"{cfg.SITE_ALIAS[site_id]}-第{replicate_id}组", fontsize=10)
            # axis.set_xlabel("调查序号")
            # axis.set_ylabel("数值")
            # # if mode_name == "full_fit":
            # #     axis.set_ylim(0, 100)
            # #     axis.set_yticks(np.arange(0, 101, 20))
            # axis.grid(alpha=0.25)

        # ===== 第4步：图例和总标题 =====
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)

        mode_cn = "全量拟合" if mode_name == "full_fit" else "交叉验证"
        fig.suptitle(f"{cfg.TARGET_PLOT_LABELS[target]}（{mode_cn}）", fontsize=16, y=0.955)

        fig.tight_layout(rect=[0, 0, 1, 0.92])

        figure_name = f"曲线图_{mode_cn}_{cfg.TARGET_LABELS[target]}.png"
        fig.savefig(cfg.FIG_DIR / figure_name, dpi=220, bbox_inches="tight")
        plt.close(fig)

def plot_cv_scatter(prediction_rows: list[dict[str, Any]]) -> None:
    filtered = [row for row in prediction_rows if row["mode"] == "cv_optimized" and row["is_baseline"] == 0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for axis, target in zip(axes, cfg.TARGET_LABELS):
        actual = []
        pred = []
        for row in filtered:
            a = row.get(f"actual_{target}")
            p = row.get(f"pred_{target}")
            if a is None or p is None:
                continue
            actual.append(a)
            pred.append(p)
        axis.scatter(actual, pred, alpha=0.8, s=28)
        if actual:
            lower = min(min(actual), min(pred))
            upper = max(max(actual), max(pred))
            axis.plot([lower, upper], [lower, upper], linestyle="--", color="gray")
        axis.set_title(cfg.TARGET_PLOT_LABELS[target])
        # axis.set_xlabel("Observed")
        # axis.set_ylabel("Predicted")
        axis.set_xlabel("观测值")
        axis.set_ylabel("预测值")
        axis.grid(alpha=0.25)
    # fig.suptitle("Leave-one-site-out CV predictions", fontsize=16)
    fig.suptitle("留一站点交叉验证预测散点图", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # fig.savefig(cfg.FIG_DIR / "cv_scatter_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(cfg.FIG_DIR / "交叉验证散点汇总图.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(corr_rows: list[dict[str, Any]]) -> None:
    ranked = defaultdict(float)
    for row in corr_rows:
        rho = row["spearman_rho"]
        if rho is None:
            continue
        ranked[row["feature"]] = max(ranked[row["feature"]], abs(float(rho)))
    selected_features = [item[0] for item in sorted(ranked.items(), key=lambda kv: kv[1], reverse=True)[:20]]
    targets = list(cfg.TARGET_LABELS.keys())
    matrix = np.full((len(selected_features), len(targets)), np.nan, dtype=np.float32)
    for i, feature in enumerate(selected_features):
        for j, target in enumerate(targets):
            for row in corr_rows:
                if row["feature"] == feature and row["target"] == target:
                    matrix[i, j] = np.nan if row["spearman_rho"] is None else float(row["spearman_rho"])
                    break
    fig, axis = plt.subplots(figsize=(10, 7))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(targets)))
    axis.set_yticks(np.arange(len(selected_features)))
    axis.set_xticklabels([cfg.TARGET_LABELS.get(name, name) for name in targets], rotation=30, ha="right")
    # axis.set_yticklabels(selected_features)  # 原英文特征名（保留，不删除）
    axis.set_yticklabels([cfg.FEATURE_LABELS.get(name, name) for name in selected_features])
    for i in range(len(selected_features)):
        for j in range(len(targets)):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    # axis.set_title("Spearman correlation heatmap")
    axis.set_title("Spearman相关性热图")
    fig.tight_layout()
    # fig.savefig(cfg.FIG_DIR / "correlation_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(cfg.FIG_DIR / "相关性热图.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# [05.6] ===== 文本摘要与报告 =====
def summarize_metrics(metrics: list[dict[str, Any]]) -> str:
    lines = []
    for row in metrics:
        r2_text = "nan" if row["r2"] != row["r2"] else f"{row['r2']:.3f}"
        lines.append(f"- {row['target_cn']}: MAE={row['mae']:.2f}, RMSE={row['rmse']:.2f}, R²={r2_text}, n={row['n']}")
    return "\n".join(lines)


def summarize_top_correlations(top_rows: list[dict[str, Any]]) -> str:
    lines = []
    for target in cfg.TARGET_LABELS:
        subset = [row for row in top_rows if row["target"] == target][:3]
        if not subset:
            continue
        joined = "；".join(
            f"{row['feature_cn']} (rho={row['spearman_rho']:.3f}, p={row['spearman_p']:.3g})"
            for row in subset
            if row["spearman_rho"] is not None and row["spearman_p"] is not None
        )
        lines.append(f"- {cfg.TARGET_LABELS[target]}: {joined}")
    return "\n".join(lines)


def summarize_top_importance(importance_rows: list[dict[str, Any]]) -> str:
    lines = []
    grouped = defaultdict(list)
    for row in importance_rows:
        grouped[row["disease"]].append(row)
    for disease_key in cfg.DISEASE_CONFIGS:
        subset = sorted(grouped[disease_key], key=lambda item: item["importance_delta_rmse"], reverse=True)[:3]
        joined = "；".join(f"{row['feature_cn']} (ΔRMSE={row['importance_delta_rmse']:.3f})" for row in subset)
        lines.append(f"- {cfg.DISEASE_CONFIGS[disease_key]['cn_name']}: {joined}")
    return "\n".join(lines)


def build_report(
    docx_text: str,
    panel_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    cv_metrics: list[dict[str, Any]],
    top_corr_rows: list[dict[str, Any]],
    importance_rows: list[dict[str, Any]],
) -> str:
    shifted = sum(1 for row in quality_rows if row["shift_days"] != 0)
    point_count = len({row["site_id"] for row in panel_rows})
    timepoint_count = len(panel_rows)
    survey_rows = len(quality_rows)
    docx_excerpt = "\n".join(docx_text.splitlines()[:6])
    report = f"""# 玉米叶斑病 LSTM 预测与相关性分析结果

## 数据与清洗
- 输入文件：`玉米生育期.docx`、`叶斑病定点调查数据.zip`
- 说明文档已完整读取，核心内容为玉米生育期阶段定义。
- 调查原始记录数：{survey_rows}
- 对齐后的监测点位数：{point_count}
- 点位-日期聚合后的时点数：{timepoint_count}
- 为与气象表年份对齐，自动执行了 `-365 天` 日期修正的记录数：{shifted}
- 因一个气象点对应多个品种/重复调查，建模时按 `序号 + 调查日期` 聚合为 11 个监测点位。

## 文档提取摘要
{docx_excerpt}

## 特征设置
- 已按你的要求移除：灰斑病抗性、大斑病抗性、白斑病抗性、经度、纬度、海拔。
- 原始气象序列共使用 {len(cfg.SEQ_FEATURES)} 个变量：风速、降水、2m比湿、2m相对湿度、温度、土壤湿度、地表温度、气压、短波辐射、土壤相对湿度、5cm土壤温度。
- 新增过程特征：近3/7/14天累计降水、连续降雨天数、降雨间歇长度、有效积温_GDD、
  近3/7/14天相对湿度均值、连续高温天数、连续低温天数、连续适温天数、
  连续高湿天数、3/7天内高湿天数、3/7天内强降雨次数、连续强降雨次数。

## 建模说明
- 模型：按病害分别训练 3 个残差式 LSTM，每个模型同时预测“发病率 + 病情指数”的相对增量。
- 时序输入：目标调查日前连续 {cfg.LOOKBACK_DAYS} 天的原始气象序列。
- 过程输入：生育期编码、调查间隔、有效积温_GDD、近3/7/14天降水、连续降雨天数、降雨间歇长度、近3/7/14天相对湿度等。
- 评估方式：留一监测点交叉验证（leave-one-site-out），并采用逐次滚动一步预测生成发病曲线。

## 交叉验证指标
{summarize_metrics(cv_metrics)}

## 相关性摘要
{summarize_top_correlations(top_corr_rows)}

## 特征重要性摘要
{summarize_top_importance(importance_rows)}
"""
    return report
