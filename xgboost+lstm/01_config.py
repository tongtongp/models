from __future__ import annotations

# [01.1] ===== 基础库导入 =====
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch

# [01.2] ===== Matplotlib 后端设置（无界面环境） =====
matplotlib.use("Agg")

# [01.2.1] ===== 中文字体设置（解决中文乱码/缺字） =====
# matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # 原默认行为（保留，不删除）
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",  # Windows 常见中文字体
    "SimHei",           # 黑体
    "SimSun",           # 宋体
    "Arial Unicode MS", # 兼容字体
    "DejaVu Sans",      # 兜底
]
matplotlib.rcParams["axes.unicode_minus"] = False

# [01.3] ===== 路径与输入文件配置 =====
ROOT = Path.cwd()
MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"
# DOCX_PATH = ROOT / "玉米生育期.docx"  # 原路径（保留，不删除）
DOCX_PATH = DATA_DIR / "玉米生育期.docx"
ZIP_PATH = ROOT / "叶斑病定点调查数据.zip"
OUT_DIR = ROOT / "results_leafspot_lstm"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = ROOT / "models"

SURVEY_FILENAME = "2025定点监测叶斑病调查数据.xlsx"
WEATHER_FILENAME = "2025年定点监测气象数据.xlsx"

# [01.4] ===== 训练参数与设备配置 =====
LOOKBACK_DAYS = 28
RANDOM_SEED = 20260306
DEVICE = torch.device("cpu")
torch.set_num_threads(1)
ALPHA_GRID = np.linspace(0.0, 1.0, 21, dtype=np.float32)

# [01.4.1] ===== LSTM倒数第二层 + XGBoost 融合参数 =====
XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_REG_LAMBDA = 1.0

MONOTONIC_TARGETS = {
    "gray_incidence",
    "gray_index",
    "blight_incidence",
    "blight_index",
    "white_incidence",
    "white_index",
}

# [01.5] ===== 标签映射 =====
TARGET_LABELS = {
    "gray_incidence": "灰斑病发病株率",
    "gray_index": "灰斑病病情指数",
    "blight_incidence": "大斑病发病株率",
    "blight_index": "大斑病病情指数",
    "white_incidence": "白斑病发病株率",
    "white_index": "白斑病病情指数",
}

# [01.5.1] ===== 风险分级规则 =====
RISK_LABELS = {
    0: "低风险",
    1: "中风险",
    2: "高风险",
}

def classify_risk(value: float | None) -> str | None:
    """将0-100连续值划分为低/中/高风险。"""
    if value is None:
        return None
    value = float(value)
    if value <= 30:
        return "低风险"
    elif value <= 70:
        return "中风险"
    else:
        return "高风险"

def risk_to_level(value: float | None) -> int | None:
    """将连续值映射为风险等级编号：低=0，中=1，高=2。"""
    if value is None:
        return None
    value = float(value)
    if value <= 30:
        return 0
    elif value <= 70:
        return 1
    else:
        return 2

def combine_risk(incidence_value: float | None, index_value: float | None) -> str | None:
    """
    综合风险：取发病株率风险和病情指数风险中更高的一级。
    """
    inc_level = risk_to_level(incidence_value)
    idx_level = risk_to_level(index_value)
    valid_levels = [x for x in [inc_level, idx_level] if x is not None]
    if not valid_levels:
        return None
    return RISK_LABELS[max(valid_levels)]

TARGET_PLOT_LABELS = {
    "gray_incidence": "灰斑病发病株率",
    "gray_index": "灰斑病病情指数",
    "blight_incidence": "大斑病发病株率",
    "blight_index": "大斑病病情指数",
    "white_incidence": "白斑病发病株率",
    "white_index": "白斑病病情指数",
}

SITE_ALIAS = {
    1: "S1 周公山镇余家村",
    2: "S2 蒙顶山茶南部加工园",
    3: "S3 宝兴县",
    4: "S4 新都-长龙",
    5: "S5 新都-红花堰",
    6: "S6 泸定-燕子沟",
    7: "S7 泸定-蔡阳坪",
    8: "S8 泸定-海子坪",
    9: "S9 德宏-农科所",
    10: "S10 德宏-江东",
    11: "S11 德宏-勐嘎",
}

FEATURE_LABELS = {
    "wind_avg": "原始_10m风速均值",
    "wind_max": "原始_10m风速最大值",
    "wind_min": "原始_10m风速最小值",

    "precip_max": "原始_24小时最大降水量",
    "precip_min": "原始_24小时最小降水量",
    "precip_sum": "原始_24小时降水量之和",

    # "humidity_avg": "原始_2m比湿",

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
    # "sample_count": "重复数",
    "stage_code": "生育期编码",
    # "days_since_first": "距首次调查天数",
    # "days_since_prev": "距上次调查天数",
    # "survey_order": "调查序号",
    "gdd_cum": "有效积温_GDD",
    "rain_3d_sum": "累计降水_3d(mm)",
    "rain_7d_sum": "累计降水_7d(mm)",
    "rain_14d_sum": "累计降水_14d(mm)",
    "rain_21d_sum": "累计降水_21d(mm)",
    "rain_28d_sum": "累计降水_28d(mm)",
    "rainy_streak_days": "连续降雨天数",
    "rain_gap_days": "降雨间歇长度(天)",

    "temp_3d_mean": "平均气温℃_3d",
    "temp_7d_mean": "平均气温℃_7d",
    "temp_14d_mean": "平均气温℃_14d",
    "temp_21d_mean": "平均气温℃_21d",
    "temp_28d_mean": "平均气温℃_28d",
    "temp_range_24h_c": "24h温差℃",

    "rh_3d_mean": "平均相对湿度_3d",
    "rh_7d_mean": "平均相对湿度_7d",
    "rh_14d_mean": "平均相对湿度_14d",
    "rh_21d_mean": "平均相对湿度_21d",
    "rh_28d_mean": "平均相对湿度_28d",
    "humidity_range_daily": "湿度日较差",
    
    "soil_rel_humidity_7d_mean": "平均土壤相对湿度_7d",
    "soil_rel_humidity_14d_mean": "平均土壤相对湿度_14d",
    "soil_rel_humidity_21d_mean": "平均土壤相对湿度_21d",
    "soil_rel_humidity_28d_mean": "平均土壤相对湿度_28d",
    

    "radiation_7d_mean": "平均短波辐射_7d",
    "radiation_28d_mean": "平均短波辐射_28d",
    "wind_7d_mean": "平均风速_7d",
    "wind_28d_mean": "平均风速_28d",

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
    "high_humidity_28d_count": "28天内高湿天数",
    "high_humidity_3d_count": "3天内高湿天数",

    "heavy_rain_3d_count": "3天内强降雨次数",
    "heavy_rain_7d_count": "7天内强降雨次数",
    "heavy_rain_28d_count": "28天内强降雨次数",
    "heavy_rain_streak_days": "连续强降雨次数",
    "max_single_day_rain_7d": "7天最大单日降雨_mm",
    "max_single_day_rain_28d": "28天最大单日降雨_mm",

    "hot_humid_streak_days": "连续高温高湿天数",
    "optimal_temp_humid_streak_days": "连续适温高湿天数",
    "weak_wind_humid_streak_days": "连续弱风高湿天数",
}

# [01.6] ===== 生育期编码与气象列映射 =====
STAGE_CODE_BASE = {
    "VE": 0.0,
    "V1": 1.0,
    "V2": 2.0,
    "V4": 4.0,
    "V6": 6.0,
    "V10": 10.0,
    "V11": 11.0,
    "V12": 12.0,
    "V14": 14.0,
    "VT": 15.0,
    "R1": 16.0,
    "R2": 17.0,
    "R3": 18.0,
    "R4": 19.0,
    "R5": 20.0,
    "R6": 21.0,
}

WEATHER_COLUMN_MAP = {
    # ===== 风速 =====
    "wind_avg": ("10m风速平均值(m/s)", lambda x: x),
    "wind_max": ("10m风速最大值(m/s)", lambda x: x),
    "wind_min": ("10m风速最小值(m/s)", lambda x: x),
    # ===== 降水 =====
    "precip_max": ("24小时内最大降水量(mm)", lambda x: x),
    "precip_min": ("24小时内最小降水量(mm)", lambda x: x),
    "precip_sum": ("24小时内降水量之和(mm)", lambda x: x),
    # ===== 比湿（仅作中间计算，不导出）=====
    "humidity_avg": ("2m比湿平均值(kg/kg)", lambda x: x),
    "humidity_max": ("2m比湿最大值(kg/kg)", lambda x: x),
    "humidity_min": ("2m比湿最小值(kg/kg)", lambda x: x),
    # ===== 气温（K -> ℃）=====
    "temp_avg_c": ("2m气温平均值(K)", lambda x: x - 273.15),
    "temp_max_c": ("2m气温最大值(K)", lambda x: x - 273.15),
    "temp_min_c": ("2m气温最小值(K)", lambda x: x - 273.15),
    # ===== 土壤湿度 =====
    "soil_moisture": ("土壤湿度(m3 m-3)", lambda x: x),
    # ===== 地表温度（K -> ℃）=====
    "surface_temp_avg_c": ("地表温度平均值(K)", lambda x: x - 273.15),
    "surface_temp_max_c": ("地表温度最大值(K)", lambda x: x - 273.15),
    "surface_temp_min_c": ("地表温度最小值(K)", lambda x: x - 273.15),
    # ===== 地面气压（Pa -> kPa）=====
    "pressure_kpa": ("地面气压平均值(Pa)", lambda x: x / 1000.0),
    "pressure_max_kpa": ("地面气压最大值(Pa)", lambda x: x / 1000.0),
    "pressure_min_kpa": ("地面气压最小值(Pa)", lambda x: x / 1000.0),
    # ===== 短波辐射 =====
    "radiation_avg": ("短波辐射平均值(W/m2)", lambda x: x),
    "radiation_max": ("短波辐射最大值(W/m2)", lambda x: x),
    "radiation_min": ("短波辐射最小值(W/m2)", lambda x: x),
    # ===== 土壤相对湿度 / 土壤温度 =====
    "soil_rel_humidity": ("0-10cm土壤相对湿度(percent)", lambda x: x),
    "soil_temp_c": ("5cm土壤温度(K)", lambda x: x - 273.15),
}

SEQ_FEATURES = [
    "wind_avg", "wind_max", "wind_min",
    "precip_max", "precip_min", "precip_sum",
    "relative_humidity", "relative_humidity_max", "relative_humidity_min",
    "temp_avg_c", "temp_max_c", "temp_min_c",
    "soil_moisture",
    "surface_temp_avg_c", "surface_temp_max_c", "surface_temp_min_c",
    "pressure_kpa", "pressure_max_kpa", "pressure_min_kpa",
    "radiation_avg", "radiation_max", "radiation_min",
    "soil_rel_humidity", "soil_temp_c",
]

BASE_MODEL_FEATURES = [
    "gdd_cum",
    "stage_code",
    "rain_21d_sum","rain_7d_sum","rain_14d_sum","rain_28d_sum",
    "rainy_streak_days","rain_gap_days",

    "temp_21d_mean","temp_7d_mean","temp_14d_mean","temp_28d_mean",
    "temp_range_24h_c",

    "rh_21d_mean","rh_7d_mean","rh_14d_mean","rh_28d_mean",
    "humidity_range_daily",

    "soil_rel_humidity_14d_mean","soil_rel_humidity_7d_mean","soil_rel_humidity_21d_mean","soil_rel_humidity_28d_mean",

    "wind_7d_mean","wind_28d_mean",
    "is_weak_wind_day",
    "weak_wind_streak_days",

    "radiation_7d_mean","radiation_28d_mean","low_radiation_streak_days",

    "hot_streak_days", "cold_streak_days","optimal_temp_streak_days",

    "high_humidity_streak_days","high_humidity_7d_count","high_humidity_28d_count",
    # "high_humidity_3d_count",

    # "heavy_rain_3d_count"
    "heavy_rain_7d_count","heavy_rain_28d_count","heavy_rain_streak_days","max_single_day_rain_7d","max_single_day_rain_28d",

    "hot_humid_streak_days","optimal_temp_humid_streak_days","weak_wind_humid_streak_days",
]

CORR_FEATURES = BASE_MODEL_FEATURES[:]
IMPORTANCE_FEATURES = SEQ_FEATURES + BASE_MODEL_FEATURES

DISEASE_CONFIGS = {
    "gray": {"prefix": "gls", "cn_name": "灰斑病", "targets": ["gray_incidence", "gray_index"]},
    "blight": {"prefix": "nlb", "cn_name": "大斑病", "targets": ["blight_incidence", "blight_index"]},
    "white": {"prefix": "wsp", "cn_name": "白斑病", "targets": ["white_incidence", "white_index"]},
}

# [01.7] ===== 站点自适应阈值与事件阈值 =====
TEMP_LOW_THRESHOLD = 18.0
TEMP_OPTIMAL_LOW = 25.0
TEMP_OPTIMAL_HIGH = 28.0
TEMP_HIGH_THRESHOLD = 28.0

HIGH_HUMIDITY_THRESHOLD = 90.0
MEDIUM_HIGH_HUMIDITY_THRESHOLD = 80.0

HEAVY_RAIN_THRESHOLD = 25.0
WEAK_WIND_THRESHOLD = 3.0
LOW_RADIATION_THRESHOLD = 100.0
# TEMP_LOW_Q = 0.20
# TEMP_HIGH_Q = 0.80

# HIGH_HUMIDITY_THRESHOLD = 85.0   # 日平均相对湿度 >= 85% 视为高湿日
# HEAVY_RAIN_THRESHOLD = 25.0      # 日降水量 >= 25 mm 视为强降雨日

# WEAK_WIND_THRESHOLD = 2.0       # 日平均风速 < 2.0 m/s 视为弱风日
# LOW_RADIATION_THRESHOLD = 100.0 # 日平均短波辐射 < 100 W/m2 视为低辐射日


# [01.8] ===== 公共初始化函数 =====
def set_global_seed(seed: int) -> None:
    """设置 random / NumPy / PyTorch 的全局随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs() -> None:
    """确保输出目录存在（结果目录、图形目录、模型目录）。"""
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
