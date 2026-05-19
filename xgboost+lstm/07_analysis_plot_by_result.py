from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


# 0. 中文乱码适配（终端 + Matplotlib）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def configure_chinese_font():
    """为 Matplotlib 自动选择可用中文字体，避免中文乱码。"""
    candidate_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidate_fonts:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    else:
        # 若系统无常见中文字体，保留默认字体，尽量不阻断流程
        print("[提示] 未检测到常见中文字体，图中中文可能显示为方块。")

    plt.rcParams["axes.unicode_minus"] = False


configure_chinese_font()

# 1. 设置你的结果文件夹路径
RESULT_DIR = Path(r"./results_leafspot_lstm")
OUTPUT_DIR = RESULT_DIR / "分析图表"

# 2. 读取“全部特征总表”
feature_df = pd.read_csv(RESULT_DIR / "全部特征总表.csv", encoding="utf-8-sig")

# # 3. 看看前5行
# print(feature_df.head())

# # 4. 看看所有列名
# print(feature_df.columns.tolist())

# =========================
# 3. 定义分箱函数
# =========================
def bin_optimal_days(x):
    if pd.isna(x):
        return None
    x = float(x)
    if x == 0:
        return "0天"
    elif 1 <= x <= 2:
        return "1–2天"
    elif 3 <= x <= 4:
        return "3–4天"
    elif 5 <= x <= 6:
        return "5–6天"
    elif 7 <= x <= 9:
        return "7–9天"
    else:
        return "≥10天"

# =========================
# 4. 生成分箱列
# =========================
bin_order = ["0天", "1–2天", "3–4天", "5–6天", "7–9天", "≥10天"]

feature_df["连续适温天数分箱"] = feature_df["连续适温天数"].apply(bin_optimal_days)
feature_df["连续适温天数分箱"] = pd.Categorical(
    feature_df["连续适温天数分箱"],
    categories=bin_order,
    ordered=True
)

# =========================
# 5. 设置目标字段
# =========================
targets = [
    "灰斑病发病株率",
    "灰斑病病情指数",
    "大斑病发病株率",
    "大斑病病情指数",
    "白斑病发病株率",
    "白斑病病情指数",
]

# =========================
# 6. 生成分箱统计表
# =========================
summary_rows = []

for b in bin_order:
    sub = feature_df[feature_df["连续适温天数分箱"] == b]

    row = {
        "连续适温天数分箱": b,
        "样本数": len(sub)
    }

    for t in targets:
        row[f"{t}_均值"] = sub[t].mean()
        row[f"{t}_中位数"] = sub[t].median()
        row[f"{t}_非零比例"] = (sub[t] > 0).mean() if len(sub) > 0 else None

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

# 保存统计表
summary_df.to_csv(OUTPUT_DIR / "连续适温天数分箱统计表.csv", index=False, encoding="utf-8-sig")
print(summary_df)

# =========================
# 7. 作图
# =========================
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, t in zip(axes, targets):
    y_values = [summary_df.loc[summary_df["连续适温天数分箱"] == b, f"{t}_均值"].values[0] for b in bin_order]

    ax.plot(bin_order, y_values, marker="o", linewidth=1.8)
    ax.set_title(t)
    ax.set_xlabel("连续适温天数分箱")
    ax.set_ylabel("均值")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "连续适温天数分箱图_六指标均值.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 8. 画“非零比例”图
# =========================
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, t in zip(axes, targets):
    y_values = [
        summary_df.loc[summary_df["连续适温天数分箱"] == b, f"{t}_非零比例"].values[0]
        for b in bin_order
    ]

    ax.plot(bin_order, y_values, marker="o", linewidth=1.8)
    ax.set_title(f"{t}（非零比例）")
    ax.set_xlabel("连续适温天数分箱")
    ax.set_ylabel("非零比例")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "连续适温天数分箱图_六指标非零比例.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 9. 画“样本数”图
# =========================
plt.figure(figsize=(8, 4))
plt.bar(summary_df["连续适温天数分箱"], summary_df["样本数"])
plt.title("连续适温天数各分箱样本数")
plt.xlabel("连续适温天数分箱")
plt.ylabel("样本数")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "连续适温天数分箱_样本数.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 10. 画“均值 + 非零比例 + 样本数”合并图
# =========================
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

for ax, t in zip(axes, targets):
    x = np.arange(len(bin_order))

    mean_values = [
        summary_df.loc[summary_df["连续适温天数分箱"] == b, f"{t}_均值"].values[0]
        for b in bin_order
    ]
    nonzero_values = [
        summary_df.loc[summary_df["连续适温天数分箱"] == b, f"{t}_非零比例"].values[0]
        for b in bin_order
    ]
    sample_counts = [
        summary_df.loc[summary_df["连续适温天数分箱"] == b, "样本数"].values[0]
        for b in bin_order
    ]

    # 左轴：样本数（柱状图）
    bars = ax.bar(x, sample_counts, alpha=0.25, label="样本数")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_order, rotation=20)
    ax.set_xlabel("连续适温天数分箱")
    ax.set_ylabel("样本数")
    ax.set_title(t)
    ax.grid(axis="y", alpha=0.3)

    # 右轴：均值 + 非零比例
    # ax2 = ax.twinx()
    # line1 = ax2.plot(x, mean_values, marker="o", linewidth=1.8, label="均值")
    # line2 = ax2.plot(x, nonzero_values, marker="s", linewidth=1.8, linestyle="--", label="非零比例")
    # ax2.set_ylabel("均值 / 非零比例")
        # 把均值归一化到 0-1，便于和非零比例一起比较
    max_mean = max(mean_values) if max(mean_values) > 0 else 1
    mean_scaled = [v / max_mean for v in mean_values]

    ax2 = ax.twinx()
    line1 = ax2.plot(x, mean_scaled, marker="o", linewidth=1.8, label="均值(归一化)")
    line2 = ax2.plot(x, nonzero_values, marker="s", linewidth=1.8, linestyle="--", label="非零比例")
    ax2.set_ylabel("归一化均值 / 非零比例")
    ax2.set_ylim(0, 1.05)

    # 把图例合并
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "连续适温天数分箱图_均值_非零比例_样本数_合并版.png", dpi=300, bbox_inches="tight")
plt.show()