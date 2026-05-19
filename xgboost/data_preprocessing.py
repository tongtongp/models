"""
玉米病害预测数据预处理模块（XGBoost 优化版）

目标：
1. 输出与在线预测一致的英文气象特征名
2. 避免病害目标逐日插值造成数据泄露
3. 增加 XGBoost 更适合的 lag / rolling / 交互特征
4. 生成病害增量目标 delta，更适合预测气象对发病变化的影响
"""


from __future__ import annotations

import os
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def _silent_print(*args, **kwargs):
    return None


print = _silent_print

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

FEATURE_NAME_MAP = {
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
    "rain_21d_sum": "累计降水_21d(mm)",
    "rain_28d_sum": "累计降水_28d(mm)",
    "rainy_streak_days": "连续降雨天数",
    "rain_gap_days": "降雨间歇长度(天)",
    "temp_range_24h_c": "24h温差℃",
    "surface_temp_range_24h_c": "地表24h温差℃",
    "pressure_range_kpa": "气压日较差(kPa)",
    "humidity_range_daily": "湿度日较差",
    "soil_rel_humidity_7d_mean": "平均土壤相对湿度_7d",
    "soil_rel_humidity_14d_mean": "平均土壤相对湿度_14d",
    "soil_rel_humidity_21d_mean": "平均土壤相对湿度_21d",
    "soil_rel_humidity_28d_mean": "平均土壤相对湿度_28d",
    "radiation_7d_mean": "平均短波辐射_7d",
    "wind_7d_mean": "平均风速_7d",

    "is_rainy_day": "是否降雨日",
    "is_high_humidity": "是否高湿日",
    "is_medium_high_humidity": "是否较高湿度日",
    "is_heavy_rain": "是否强降雨日",
    "is_hot_day": "是否高温日",
    "is_cold_day": "是否低温日",
    "is_optimal_temp_day": "是否适温日",
    "is_low_radiation_day": "是否寡照日",
    "hot_humid_flag": "是否高温高湿日",
    "optimal_temp_humid_flag": "是否适温高湿日",
    "weak_wind_humid_flag": "是否弱风高湿日",
    "disease_suitable_day": "是否适宜发病日",

    "humid_temp_interaction": "温湿交互",
    "rain_humidity_interaction": "降水-湿度交互",
    "weak_wind_humidity_interaction": "弱风-湿度交互",

    "gdd_daily": "当日有效积温_GDD",
    "soil_temp_c": "原始_5cm土壤温度(℃)",
    "is_weak_wind_day": "是否弱风日",
    "weak_wind_streak_days": "弱风日连续天数",
    "is_low_radiation_day_streak_days": "寡照连续天数",
    "hot_streak_days": "连续高温天数",
    "cold_streak_days": "连续低温天数",
    "optimal_temp_streak_days": "连续适温天数",
    "high_humidity_streak_days": "连续高湿天数",
    "medium_high_humidity_streak_days": "连续较高湿度天数",
    "high_humidity_7d_count": "7天内高湿天数",
    "high_humidity_3d_count": "3天内高湿天数",
    "heavy_rain_3d_count": "3天内强降雨次数",
    "heavy_rain_7d_count": "7天内强降雨次数",
    "heavy_rain_streak_days": "连续强降雨次数",
    "max_single_day_rain_7d": "7天最大单日降雨_mm",
    "hot_humid_streak_days": "连续高温高湿天数",
    "optimal_temp_humid_streak_days": "连续适温高湿天数",
    "weak_wind_humid_streak_days": "连续弱风高湿天数",
}


class CornDiseaseDataPreprocessor:
    """玉米病害数据预处理器（XGBoost 优化版）"""

    def __init__(self, disease_file: str, weather_file: str):
        self.disease_file = disease_file
        self.weather_file = weather_file
        self.disease_data: pd.DataFrame | None = None
        self.weather_data: pd.DataFrame | None = None
        self.merged_data: pd.DataFrame | None = None

    @staticmethod
    def _map_feature_name(name: str) -> str:
        if name in FEATURE_NAME_MAP:
            return FEATURE_NAME_MAP[name]

        base_map = {
            "temp_avg_c": "平均气温",
            "temp_max_c": "最高气温",
            "temp_min_c": "最低气温",
            "relative_humidity": "相对湿度",
            "wind_avg": "风速",
            "radiation_avg": "短波辐射",
            "soil_rel_humidity": "土壤相对湿度",
            "soil_moisture": "土壤湿度",
            "soil_temp_c": "土壤温度",
            "temp_range_24h_c": "24h温差",
            "humidity_range_daily": "湿度日较差",
        }
        stat_map = {
            "mean": "平均",
            "max": "最大",
            "min": "最小",
            "std": "标准差",
        }

        m = re.match(r"(.+)_([0-9]+)d_(mean|max|min|std)", name)
        if m:
            base, days, stat = m.group(1), m.group(2), m.group(3)
            if base in base_map:
                return f"{base_map[base]}_{days}d{stat_map[stat]}"

        m = re.match(r"(.+)_([0-9]+)d_(sum|max|mean|count)", name)
        if m:
            base, days, stat = m.group(1), m.group(2), m.group(3)
            base_cn = {
                "rain": "降水",
                "heavy_rain": "强降雨",
                "high_humidity": "高湿",
                "disease_suitable": "适宜发病",
                "gdd": "有效积温",
            }.get(base)
            if base_cn:
                stat_cn = {
                    "sum": "累计",
                    "max": "最大",
                    "mean": "平均",
                    "count": "次数",
                }[stat]
                if stat == "count":
                    return f"{days}天内{base_cn}{stat_cn}"
                return f"{base_cn}_{days}d{stat_cn}"

        m = re.match(r"(.+)_streak_days", name)
        if m:
            base = m.group(1)
            base_cn = {
                "is_rainy_day": "降雨",
                "is_weak_wind_day": "弱风",
                "is_low_radiation_day": "寡照",
                "is_hot_day": "高温",
                "is_cold_day": "低温",
                "is_optimal_temp_day": "适温",
                "is_high_humidity": "高湿",
                "is_medium_high_humidity": "较高湿度",
                "is_heavy_rain": "强降雨",
                "hot_humid_flag": "高温高湿",
                "optimal_temp_humid_flag": "适温高湿",
                "weak_wind_humid_flag": "弱风高湿",
                "disease_suitable_day": "适宜发病",
            }.get(base)
            if base_cn:
                return f"连续{base_cn}天数"

        return name

    @staticmethod
    def _make_unique_labels(labels: list[str]) -> list[str]:
        counts: dict[str, int] = {}
        out: list[str] = []
        for lbl in labels:
            if lbl not in counts:
                counts[lbl] = 0
                out.append(lbl)
            else:
                counts[lbl] += 1
                out.append(f"{lbl}({counts[lbl]})")
        return out

    # =========================
    # 基础工具函数
    # =========================

    def _parse_any_date(self, value: Any) -> pd.Timestamp:
        if pd.isna(value):
            return pd.NaT

        if isinstance(value, (int, float, np.integer, np.floating)):
            num = float(value)
            if np.isfinite(num):
                s_num = str(int(abs(num)))

                if len(s_num) >= 8 and s_num[:4].isdigit():
                    year = int(s_num[:4])
                    if 1900 <= year <= 2100:
                        try:
                            return pd.to_datetime(
                                s_num[:8],
                                format="%Y%m%d",
                                errors="raise"
                            ).normalize()
                        except Exception:
                            pass

                if 10000 < num < 80000:
                    return (
                        pd.Timestamp("1899-12-30") + pd.Timedelta(days=num)
                    ).normalize()

        s = str(value).strip()
        if s == "" or s.lower() in {"nan", "none", "nat"}:
            return pd.NaT

        digits = "".join(ch for ch in s if ch.isdigit())

        if len(digits) >= 8:
            try:
                return pd.to_datetime(
                    digits[:8],
                    format="%Y%m%d",
                    errors="raise"
                ).normalize()
            except Exception:
                pass

        if digits.isdigit() and 4 <= len(digits) <= 6:
            try:
                num = float(digits)
                if num > 10000:
                    return (
                        pd.Timestamp("1899-12-30") + pd.Timedelta(days=num)
                    ).normalize()
            except Exception:
                pass

        dt = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt):
            return pd.NaT

        if dt.year < 2000:
            return pd.NaT

        return dt.normalize()

    def _find_col(self, df: pd.DataFrame, keywords: str | list[str]) -> pd.Series:
        if isinstance(keywords, str):
            keywords = [keywords]

        for kw in keywords:
            if kw in df.columns:
                return df[kw]

        for kw in keywords:
            kw_clean = kw.replace(" ", "")

            for col in df.columns:
                if kw_clean in str(col).replace(" ", ""):
                    return df[col]

            kw_core = kw.split("(")[0].strip()
            if kw_core:
                for col in df.columns:
                    if kw_core in str(col):
                        return df[col]

        print(f"⚠️ 警告: 找不到匹配列 {keywords}，该特征置为 NaN")
        return pd.Series(np.nan, index=df.index, name=keywords[0])

    def _to_numeric_keep_nan(self, s: pd.Series) -> pd.Series:
        return pd.to_numeric(
            s.replace(["-", "无", "None", "nan", "", " "], np.nan),
            errors="coerce"
        )

    @staticmethod
    def _safe_div(a, b):
        return np.where(np.abs(b) < 1e-9, np.nan, a / b)

    @staticmethod
    def kelvin_to_celsius(kelvin_temp):
        return kelvin_temp - 273.15

    def calculate_relative_humidity(self, specific_humidity, temperature_k, pressure_pa):
        temp_c = self.kelvin_to_celsius(temperature_k)

        es = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))

        q = pd.to_numeric(specific_humidity, errors="coerce")
        p = pd.to_numeric(pressure_pa, errors="coerce")

        mixing_ratio = q / (1 - q)
        e = (mixing_ratio * p / 100) / (0.622 + mixing_ratio)
        rh = (e / es) * 100

        return np.clip(rh, 0, 100)

    def _encode_growth_stage(self, stage: Any) -> float:
        if pd.isna(stage):
            return np.nan

        s = str(stage).strip().upper()
        if s == "":
            return np.nan

        if s in {"VE", "V.E"}:
            return 0.5

        s = (
            s.replace(" ", "")
            .replace("／", "/")
            .replace("－", "-")
            .replace("—", "-")
        )

        def parse_stage(prefix: str, base: float) -> float | None:
            if not s.startswith(prefix):
                return None

            tail = s[len(prefix):]

            if tail == "T":
                return base + 10.0 if prefix == "V" else base

            if "-" in tail:
                parts = [p for p in tail.split("-") if p]
                values = []
                for part in parts:
                    if part.isdigit():
                        values.append(base + float(part))
                return float(np.mean(values)) if values else None

            if tail.isdigit():
                return base + float(tail)

            return None

        stage_value = parse_stage("V", 0.0)
        if stage_value is not None:
            return float(stage_value)

        stage_value = parse_stage("R", 20.0)
        if stage_value is not None:
            return float(stage_value)

        digits = re.findall(r"\d+", s)
        if digits:
            nums = [float(x) for x in digits]
            return float(np.mean(nums))

        return np.nan

    # =========================
    # 加载病害数据
    # =========================

    def load_disease_data(self) -> pd.DataFrame:
        print("正在加载病害调查数据...")

        excel_file = pd.ExcelFile(self.disease_file)
        all_disease_data = []

        for sheet_name in excel_file.sheet_names:
            if "定点监测站点" in sheet_name or "统计" in sheet_name or "说明" in sheet_name:
                continue

            df = pd.read_excel(self.disease_file, sheet_name=sheet_name)
            df["__sheet__"] = sheet_name

            if "地点" in df.columns or "时间" in df.columns:
                all_disease_data.append(df)

        if all_disease_data:
            self.disease_data = pd.concat(all_disease_data, ignore_index=True)
        else:
            self.disease_data = pd.read_excel(self.disease_file)

        self.disease_data.columns = [
            str(col).strip().replace("\n", "") for col in self.disease_data.columns
        ]

        # 识别并统一品种列名
        if "品种" not in self.disease_data.columns:
            variety_col = None
            for col in self.disease_data.columns:
                col_str = str(col).replace(" ", "")
                if "品种" in col_str or "品系" in col_str:
                    variety_col = col
                    break
            if variety_col is not None:
                self.disease_data = self.disease_data.rename(columns={variety_col: "品种"})

        if "地点" in self.disease_data.columns:
            mask = self.disease_data["地点"].astype(str).str.contains(
                "定点监测站点|统计|汇总",
                na=False
            )
            self.disease_data = self.disease_data[~mask].copy()

        # 如果品种列存在但大量为空，尝试用其他候选列补齐（不做跨行填充，避免误用上方品种）
        if "品种" in self.disease_data.columns:
            candidate_cols = ["品种"]
            for col in self.disease_data.columns:
                col_str = str(col).replace(" ", "")
                if ("品种" in col_str or "品系" in col_str) and col not in candidate_cols:
                    candidate_cols.append(col)

            def _clean_variety(s: pd.Series) -> pd.Series:
                return (
                    s.replace(["", " ", "-", "无", "None", "nan"], np.nan)
                    .astype("string")
                    .str.strip()
                )

            for col in candidate_cols:
                self.disease_data[col] = _clean_variety(self.disease_data[col])

            for col in candidate_cols[1:]:
                self.disease_data["品种"] = self.disease_data["品种"].combine_first(self.disease_data[col])

        if "时间" in self.disease_data.columns:
            self.disease_data["时间"] = self.disease_data["时间"].apply(self._parse_any_date)
            self.disease_data = self.disease_data[self.disease_data["时间"].notna()].copy()

        if "品种" in self.disease_data.columns:
            self.disease_data["品种"] = (
                self.disease_data["品种"]
                .replace(["", " ", "-", "无", "None", "nan"], np.nan)
                .astype("string")
                .str.strip()
                .fillna("未知品种")
            )

        if "__sheet__" in self.disease_data.columns:
            self.disease_data = self.disease_data.drop(columns=["__sheet__"])

        target_cols = self.get_target_columns()

        for col in target_cols:
            if col in self.disease_data.columns:
                self.disease_data[col] = self._to_numeric_keep_nan(self.disease_data[col])

        print(f"病害数据加载完成，共 {len(self.disease_data)} 条记录")
        return self.disease_data

    # =========================
    # 加载气象数据
    # =========================

    def load_weather_data(self) -> pd.DataFrame:
        print("正在加载气象数据...")

        excel_file = pd.ExcelFile(self.weather_file)
        all_weather_data = []

        for sheet_name in excel_file.sheet_names:
            if sheet_name == "定点监测站点" or "站点" in sheet_name or "说明" in sheet_name:
                continue

            df = pd.read_excel(self.weather_file, sheet_name=sheet_name)
            df.columns = [str(col).strip().replace("\n", "") for col in df.columns]

            df["地点"] = sheet_name

            date_col = None
            for c in ["date", "日期", "时间"]:
                if c in df.columns:
                    date_col = c
                    break

            if date_col is None:
                print(f"⚠️ 工作表 {sheet_name} 未找到日期列，跳过")
                continue

            df["date"] = df[date_col].apply(self._parse_any_date)
            df = df[df["date"].notna()].copy()

            all_weather_data.append(df)

        if not all_weather_data:
            raise ValueError("未读取到有效气象数据，请检查 Excel 工作表和日期列")

        self.weather_data = pd.concat(all_weather_data, ignore_index=True)

        self.weather_data = (
            self.weather_data
            .sort_values(["地点", "date"]) 
            .drop_duplicates(subset=["地点", "date"], keep="last")
            .reset_index(drop=True)
        )

        print(f"气象数据加载完成，共 {len(self.weather_data)} 条记录")
        return self.weather_data

    # =========================
    # 生成基础气象特征
    # =========================

    def generate_derived_features(self) -> pd.DataFrame:
        print("正在生成基础气象特征...")

        if self.weather_data is None:
            raise ValueError("请先加载气象数据")

        df = self.weather_data.copy()
        df.columns = [str(col).strip().replace("\n", "") for col in df.columns]

        wind_avg = self._find_col(df, ["10m风速平均值 (m/s)", "10m风速平均值"])
        wind_max = self._find_col(df, ["10m风速最大值 (m/s)", "10m风速最大值"])
        wind_min = self._find_col(df, ["10m风速最小值 (m/s)", "10m风速最小值"])

        precip_sum = self._find_col(df, ["24小时内降水量之和 (mm)", "24小时内降水量之和"])
        precip_max = self._find_col(df, ["24小时内降水量最大值 (mm)", "24小时内降水量最大值", "24小时内最大降水量"])
        precip_min = self._find_col(df, ["24小时内降水量最小值 (mm)", "24小时内降水量最小值", "24小时内最小降水量"])

        temp_avg_k = self._find_col(df, ["2m气温平均值 (K)", "2m气温平均值"])
        temp_max_k = self._find_col(df, ["2m气温最大值 (K)", "2m气温最大值"])
        temp_min_k = self._find_col(df, ["2m气温最小值 (K)", "2m气温最小值"])

        surface_temp_avg_k = self._find_col(df, ["地表温度平均值 (K)", "地表温度平均值"])
        surface_temp_max_k = self._find_col(df, ["地表温度最大值 (K)", "地表温度最大值"])
        surface_temp_min_k = self._find_col(df, ["地表温度最小值 (K)", "地表温度最小值"])

        pressure_avg_pa = self._find_col(df, ["地面气压平均值 (Pa)", "地面气压平均值"])
        pressure_max_pa = self._find_col(df, ["地面气压最大值 (Pa)", "地面气压最大值"])
        pressure_min_pa = self._find_col(df, ["地面气压最小值 (Pa)", "地面气压最小值"])

        q_avg = self._find_col(df, ["2m比湿平均值 (kg/kg)", "2m比湿平均值"])
        q_max = self._find_col(df, ["2m比湿最大值 (kg/kg)", "2m比湿最大值"])
        q_min = self._find_col(df, ["2m比湿最小值 (kg/kg)", "2m比湿最小值"])

        soil_rel_humidity = self._find_col(df, ["0-10cm土壤相对湿度(percent)", "0-10cm土壤相对湿度", "土壤湿度"])
        soil_temp_k = self._find_col(df, ["5cm土壤温度 (K)", "5cm土壤温度"])
        soil_moisture = self._find_col(df, ["0-10cm土壤湿度(kg/m2)", "0-10cm土壤湿度", "土壤水分"])

        radiation_avg = self._find_col(df, ["短波辐射平均值(W/m2)", "短波辐射平均值", "短波辐射"])
        radiation_max = self._find_col(df, ["短波辐射最大值(W/m2)", "短波辐射最大值"])
        radiation_min = self._find_col(df, ["短波辐射最小值(W/m2)", "短波辐射最小值"])

        df["wind_avg"] = self._to_numeric_keep_nan(wind_avg)
        df["wind_max"] = self._to_numeric_keep_nan(wind_max)
        df["wind_min"] = self._to_numeric_keep_nan(wind_min)

        df["precip_sum"] = self._to_numeric_keep_nan(precip_sum)
        df["precip_max"] = self._to_numeric_keep_nan(precip_max)
        df["precip_min"] = self._to_numeric_keep_nan(precip_min)

        df["temp_avg_c"] = self._to_numeric_keep_nan(self.kelvin_to_celsius(temp_avg_k))
        df["temp_max_c"] = self._to_numeric_keep_nan(self.kelvin_to_celsius(temp_max_k))
        df["temp_min_c"] = self._to_numeric_keep_nan(self.kelvin_to_celsius(temp_min_k))

        df["surface_temp_avg_c"] = self._to_numeric_keep_nan(self.kelvin_to_celsius(surface_temp_avg_k))
        df["surface_temp_max_c"] = self._to_numeric_keep_nan(self.kelvin_to_celsius(surface_temp_max_k))
        df["surface_temp_min_c"] = self._to_numeric_keep_nan(self.kelvin_to_celsius(surface_temp_min_k))

        df["pressure_kpa"] = self._to_numeric_keep_nan(pressure_avg_pa) / 1000.0
        df["pressure_max_kpa"] = self._to_numeric_keep_nan(pressure_max_pa) / 1000.0
        df["pressure_min_kpa"] = self._to_numeric_keep_nan(pressure_min_pa) / 1000.0

        df["relative_humidity"] = self._to_numeric_keep_nan(
            pd.Series(
                self.calculate_relative_humidity(q_avg, temp_avg_k, pressure_avg_pa),
                index=df.index
            )
        )

        df["relative_humidity_max"] = self._to_numeric_keep_nan(
            pd.Series(
                self.calculate_relative_humidity(q_max, temp_max_k, pressure_max_pa),
                index=df.index
            )
        )

        df["relative_humidity_min"] = self._to_numeric_keep_nan(
            pd.Series(
                self.calculate_relative_humidity(q_min, temp_min_k, pressure_min_pa),
                index=df.index
            )
        )

        df["soil_rel_humidity"] = self._to_numeric_keep_nan(soil_rel_humidity)
        df["soil_temp_c"] = self._to_numeric_keep_nan(self.kelvin_to_celsius(soil_temp_k))
        df["soil_moisture"] = self._to_numeric_keep_nan(soil_moisture)

        df["radiation_avg"] = self._to_numeric_keep_nan(radiation_avg)
        df["radiation_max"] = self._to_numeric_keep_nan(radiation_max)
        df["radiation_min"] = self._to_numeric_keep_nan(radiation_min)

        df = df.sort_values(["地点", "date"]).reset_index(drop=True)

        df = self.add_xgb_weather_features(df)

        if "品种" not in df.columns:
            df["品种"] = "未知品种"

        self.weather_data = df.reset_index(drop=True)

        print("XGBoost 气象派生特征生成完成")
        return self.weather_data

    # =========================
    # XGBoost 专用气象特征
    # =========================

    def add_xgb_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("正在生成 XGBoost lag / rolling / 交互特征...")

        df = df.sort_values(["地点", "date"]).copy()
        grouped = df.groupby("地点", group_keys=False)

        # 不要把缺失气象值统一填 0（0 会被模型当成真实天气），保留 NaN，XGBoost 可处理 NaN
        # 如果业务上确实需要将部分缺失视作 0，请在外部显式处理

        # 日变化范围
        df["temp_range_24h_c"] = df["temp_max_c"] - df["temp_min_c"]
        df["humidity_range_daily"] = df["relative_humidity_max"] - df["relative_humidity_min"]
        df["surface_temp_range_24h_c"] = df["surface_temp_max_c"] - df["surface_temp_min_c"]
        df["pressure_range_kpa"] = df["pressure_max_kpa"] - df["pressure_min_kpa"]

        # 基础阈值特征
        df["is_weak_wind_day"] = (df["wind_avg"] < 3).astype(int)
        df["is_rainy_day"] = (df["precip_sum"] > 0).astype(int)
        df["is_high_humidity"] = (df["relative_humidity"] >= 90).astype(int)
        df["is_medium_high_humidity"] = (df["relative_humidity"] >= 80).astype(int)
        df["is_heavy_rain"] = (df["precip_sum"] >= 25).astype(int)
        df["is_hot_day"] = (df["temp_avg_c"] > 28).astype(int)
        df["is_cold_day"] = (df["temp_avg_c"] < 18).astype(int)
        df["is_optimal_temp_day"] = (
            (df["temp_avg_c"] >= 20) & (df["temp_avg_c"] <= 30)
        ).astype(int)
        df["is_low_radiation_day"] = (df["radiation_avg"] < 100).astype(int)

        # 交互特征
        df["humid_temp_interaction"] = df["relative_humidity"] * df["temp_avg_c"]
        df["rain_humidity_interaction"] = df["precip_sum"] * df["relative_humidity"]
        df["weak_wind_humidity_interaction"] = self._safe_div(
            df["relative_humidity"],
            df["wind_avg"] + 0.1
        )

        df["hot_humid_flag"] = (
            (df["temp_avg_c"] > 28) & (df["relative_humidity"] >= 90)
        ).astype(int)

        df["optimal_temp_humid_flag"] = (
            (df["temp_avg_c"].between(20, 30)) &
            (df["relative_humidity"] >= 80)
        ).astype(int)

        df["weak_wind_humid_flag"] = (
            (df["wind_avg"] < 3) &
            (df["relative_humidity"] >= 80)
        ).astype(int)

        df["disease_suitable_day"] = (
            (df["temp_avg_c"].between(20, 30)) &
            (df["relative_humidity"] >= 80) &
            (df["wind_avg"] <= 4)
        ).astype(int)

        # GDD 积温
        df["gdd_daily"] = np.maximum(df["temp_avg_c"] - 10.0, 0.0)
        df["gdd_cum"] = grouped["gdd_daily"].cumsum()

        base_cols = [
            "temp_avg_c",
            "temp_max_c",
            "temp_min_c",
            "relative_humidity",
            "precip_sum",
            "wind_avg",
            "radiation_avg",
            "soil_rel_humidity",
            "soil_temp_c",
            "soil_moisture",
            "temp_range_24h_c",
            "humidity_range_daily",
        ]

        # rolling 均值、最大、最小、标准差
        rolling_cols = [
            "temp_avg_c",
            "relative_humidity",
            "wind_avg",
            "radiation_avg",
            "soil_rel_humidity",
            "soil_moisture",
            "temp_range_24h_c",
            "humidity_range_daily",
        ]

        for col in rolling_cols:
            if col in df.columns:
                for w in [3, 7, 14, 21, 28]:
                    min_p = max(2, w // 3)

                    df[f"{col}_{w}d_mean"] = grouped[col].transform(
                        lambda s, w=w, min_p=min_p: s.rolling(w, min_periods=min_p).mean()
                    )
                    df[f"{col}_{w}d_max"] = grouped[col].transform(
                        lambda s, w=w, min_p=min_p: s.rolling(w, min_periods=min_p).max()
                    )
                    df[f"{col}_{w}d_min"] = grouped[col].transform(
                        lambda s, w=w, min_p=min_p: s.rolling(w, min_periods=min_p).min()
                    )
                    df[f"{col}_{w}d_std"] = grouped[col].transform(
                        lambda s, w=w, min_p=min_p: s.rolling(w, min_periods=min_p).std()
                    )

        # 降雨窗口
        for w in [3, 7, 14, 21, 28]:
            min_p = max(2, w // 3)

            df[f"rain_{w}d_sum"] = grouped["precip_sum"].transform(
                lambda s, w=w, min_p=min_p: s.rolling(w, min_periods=min_p).sum()
            )
            df[f"rain_{w}d_max"] = grouped["precip_sum"].transform(
                lambda s, w=w, min_p=min_p: s.rolling(w, min_periods=min_p).max()
            )
            df[f"rain_{w}d_mean"] = grouped["precip_sum"].transform(
                lambda s, w=w, min_p=min_p: s.rolling(w, min_periods=min_p).mean()
            )
            df[f"rain_{w}d_count"] = grouped["is_rainy_day"].transform(
                lambda s, w=w, min_p=min_p: s.rolling(w, min_periods=min_p).sum()
            )
            df[f"heavy_rain_{w}d_count"] = grouped["is_heavy_rain"].transform(
                lambda s, w=w, min_p=min_p: s.rolling(w, min_periods=min_p).sum()
            )
            df[f"high_humidity_{w}d_count"] = grouped["is_high_humidity"].transform(
                lambda s, w=w, min_p=min_p: s.rolling(w, min_periods=min_p).sum()
            )
            df[f"disease_suitable_{w}d_count"] = grouped["disease_suitable_day"].transform(
                lambda s, w=w, min_p=min_p: s.rolling(w, min_periods=min_p).sum()
            )
            df[f"gdd_{w}d_sum"] = grouped["gdd_daily"].transform(
                lambda s, w=w, min_p=min_p: s.rolling(w, min_periods=min_p).sum()
            )

        # 连续天数特征
        def _streak(s: pd.Series) -> pd.Series:
            vals = s.fillna(0).astype(int).to_numpy()
            out = np.zeros(len(vals), dtype=np.int32)
            count = 0

            for i, v in enumerate(vals):
                count = count + 1 if v == 1 else 0
                out[i] = count

            return pd.Series(out, index=s.index)

        streak_cols = [
            "is_rainy_day",
            "is_weak_wind_day",
            "is_low_radiation_day",
            "is_hot_day",
            "is_cold_day",
            "is_optimal_temp_day",
            "is_high_humidity",
            "is_medium_high_humidity",
            "is_heavy_rain",
            "hot_humid_flag",
            "optimal_temp_humid_flag",
            "weak_wind_humid_flag",
            "disease_suitable_day",
        ]

        for col in streak_cols:
            df[f"{col}_streak_days"] = grouped[col].transform(_streak)

        # 无雨间隔
        def _rain_gap(s: pd.Series) -> pd.Series:
            vals = (s.fillna(0) <= 0).astype(int).to_numpy()
            out = np.zeros(len(vals), dtype=np.int32)
            count = 0

            for i, v in enumerate(vals):
                count = count + 1 if v == 1 else 0
                out[i] = count

            return pd.Series(out, index=s.index)

        df["rain_gap_days"] = grouped["precip_sum"].transform(_rain_gap)

        return df

    def save_feature_target_correlations(
        self,
        output_dir: str = "outputs/figures/feature_importance",
        top_n: int = 20,
        targets: list[str] | None = None,
        use_delta: bool = True,
    ) -> list[tuple[str, str, str]]:
        """对所有数值特征与病害目标做相关性分析，并保存可视化图片与 CSV。

        - 使用 Spearman 相关性（对异常值和非线性更稳健）
        - 优先使用 *_delta 目标（病害增量），若不存在则使用原始目标列
        - 返回列表 (target, csv_path, png_path)
        """
        if self.merged_data is None:
            raise ValueError("请先执行 merge_data_by_location() 并生成 `merged_data`")

        os.makedirs(output_dir, exist_ok=True)

        feature_cols = self.get_feature_columns()

        # 选择目标列：优先 *_delta（可关闭）或显式传入 targets
        if targets is None:
            if use_delta:
                targets = [c for c in self.merged_data.columns if str(c).endswith("_delta")]
            if not targets:
                targets = self.get_target_columns()

        corr_matrix = pd.DataFrame(index=feature_cols)

        results: list[tuple[str, str, str]] = []

        for target in targets:
            ser_t = self.merged_data[target]
            corrs = {}
            mis = {}
            for f in feature_cols:
                # Spearman: drop NaN pairwise
                try:
                    pair = self.merged_data[[f, target]].dropna()
                    if len(pair) < 10:
                        corr_val = np.nan
                    else:
                        corr_val = pair[f].corr(pair[target], method="spearman")
                except Exception:
                    corr_val = np.nan
                corrs[f] = corr_val

                # Mutual information: drop NaN, require at least 50 samples for stability
                try:
                    mi_pair = self.merged_data[[f, target]].dropna()
                    if len(mi_pair) < 50:
                        mi_val = np.nan
                    else:
                        # mutual_info_regression expects 2D X
                        mi_val = mutual_info_regression(
                            mi_pair[[f]].values, mi_pair[target].values, random_state=0
                        )[0]
                except Exception:
                    mi_val = np.nan
                mis[f] = mi_val

            corr_series = pd.Series(corrs)
            mi_series = pd.Series(mis)
            corr_matrix[target] = corr_series

            # 取绝对值排序的 top_n
            top_features = corr_series.abs().dropna().sort_values(ascending=False).head(top_n)
            top_idx = top_features.index.tolist()

            combined_df = pd.DataFrame({
                "feature": top_idx,
                "spearman": corr_series.loc[top_idx].values,
                "abs_spearman": corr_series.loc[top_idx].abs().values,
                "mutual_info": mi_series.loc[top_idx].values,
            })

            # 规范化 mutual_info 便于可视化（对 top 部分）
            if combined_df["mutual_info"].notna().any():
                max_mi = combined_df["mutual_info"].abs().max()
                if max_mi > 0:
                    combined_df["mutual_info_norm"] = combined_df["mutual_info"] / max_mi
                else:
                    combined_df["mutual_info_norm"] = combined_df["mutual_info"]
            else:
                combined_df["mutual_info_norm"] = np.nan

            csv_path = os.path.join(output_dir, f"feature_correlation_{target}.csv")
            combined_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

            # 绘图（barh: spearman + MI 点）
            vals = combined_df.set_index("feature")
            vals = vals.sort_values("spearman")
            fig_h = max(4.5, len(vals) * 0.42)
            fig, ax = plt.subplots(figsize=(11, fig_h))
            colors = ["#2F80ED" if v >= 0 else "#F2994A" for v in vals["spearman"]]
            display_index = [self._map_feature_name(x) for x in vals.index]
            display_index = self._make_unique_labels(display_index)
            bars = ax.barh(display_index, vals["spearman"], color=colors, alpha=0.9)
            ax.set_xlabel("Spearman 相关系数")
            ax.set_title(f"{target} 相关性 Top {len(vals)} 特征")
            ax.grid(axis="x", linestyle="--", alpha=0.25)

            max_abs = float(np.nanmax(np.abs(vals["spearman"].values))) if len(vals) else 1.0
            pad = max(0.05, max_abs * 0.15)
            ax.set_xlim(-max_abs - pad, max_abs + pad)

            for bar, v in zip(bars, vals["spearman"]):
                xpos = v + (0.01 if v >= 0 else -0.01)
                ax.text(
                    xpos,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}",
                    va="center",
                    ha=("left" if v >= 0 else "right"),
                    fontsize=9,
                    color="#2C3E50",
                )

            # overlay mutual info as points
            if vals["mutual_info_norm"].notna().any():
                ax2 = ax.twiny()
                ax2.plot(vals["mutual_info_norm"], display_index, "o", color="#111111", markersize=4)
                ax2.set_xlabel("互信息（归一化）")
                ax2.grid(False)

            plt.tight_layout()
            png_path = os.path.join(output_dir, f"feature_correlation_{target}.png")
            plt.savefig(png_path, dpi=220, bbox_inches="tight")
            plt.close()

            results.append((target, csv_path, png_path))

        # 保存总体排行（按各目标绝对相关性均值）
        mean_abs = corr_matrix.abs().mean(axis=1).dropna().sort_values(ascending=False)
        overall_top = mean_abs.head(top_n)
        overall_df = pd.DataFrame({"feature": overall_top.index, "mean_abs_spearman": overall_top.values})
        overall_csv = os.path.join(output_dir, "feature_importance_overall_by_spearman.csv")
        overall_df.to_csv(overall_csv, index=False, encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(9, max(4, len(overall_top) * 0.42)))
        overall_top.sort_values().plot(kind="barh", ax=ax, color="#27AE60", alpha=0.9)
        ax.set_xlabel("平均 |Spearman|（跨目标）")
        ax.set_title("总体特征重要性（Spearman 绝对相关均值）")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        plt.tight_layout()
        overall_png = os.path.join(output_dir, "feature_importance_overall_by_spearman.png")
        plt.savefig(overall_png, dpi=220, bbox_inches="tight")
        plt.close()

        # Spearman 热力图（选取整体 top_n 特征）
        heat_features = overall_top.index.tolist()
        heat_df = corr_matrix.loc[heat_features].copy()
        heat_df = heat_df.reindex(heat_features)
        heat_df = heat_df[sorted(heat_df.columns)]

        fig_size = max(7.5, 0.55 * max(len(heat_df.columns), len(heat_df.index)))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(heat_df.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(range(len(heat_df.columns)))
        ax.set_xticklabels(heat_df.columns, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(heat_df.index)))
        display_heat_idx = [self._map_feature_name(x) for x in heat_df.index]
        display_heat_idx = self._make_unique_labels(display_heat_idx)
        ax.set_yticklabels(display_heat_idx, fontsize=9)
        ax.set_title("Spearman相关性热图", fontsize=12, pad=10)

        # 标注数值
        for i in range(heat_df.shape[0]):
            for j in range(heat_df.shape[1]):
                val = heat_df.iat[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="#1F2D3D")

        # 网格线
        ax.set_xticks(np.arange(-.5, len(heat_df.columns), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(heat_df.index), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
        cbar.set_label("Spearman 相关系数")

        plt.tight_layout()
        heat_png = os.path.join(output_dir, f"spearman_heatmap_top{len(heat_df)}.png")
        plt.savefig(heat_png, dpi=240, bbox_inches="tight")
        plt.close()

        results.insert(0, ("overall", overall_csv, overall_png))
        return results

    # =========================
    # 地点模糊匹配
    # =========================

    def _fuzzy_match_location(self, location: Any, available_locations: list[Any]) -> Any:
        if location in available_locations:
            return location

        import difflib

        location_str = str(location).strip()

        def get_core_name(name: str) -> str:
            s = re.sub(r"^\d+[\.、\s]*", "", str(name).strip())

            prefixes = [
                "四川省", "四川",
                "云南省", "云南",
                "雅安市", "雅安",
                "成都市", "成都",
                "甘孜州", "甘孜",
                "德宏州", "德宏",
            ]

            for prefix in prefixes:
                if s.startswith(prefix):
                    s = s[len(prefix):]

            return s

        core_loc = get_core_name(location_str)

        for loc in available_locations:
            loc_str = str(loc).strip()
            core_avail = get_core_name(loc_str)

            if len(core_loc) >= 2 and len(core_avail) >= 2:
                if core_loc in core_avail or core_avail in core_loc:
                    return loc

        for loc in available_locations:
            loc_str = str(loc).strip()
            if location_str in loc_str or loc_str in location_str:
                return loc

        matches = difflib.get_close_matches(
            location_str,
            [str(l).strip() for l in available_locations],
            n=1,
            cutoff=0.3
        )

        if matches:
            for loc in available_locations:
                if str(loc).strip() == matches[0]:
                    return loc

        return location

    # =========================
    # 合并病害与气象数据
    # =========================

    def get_target_columns(self) -> list[str]:
        return [
            "灰斑病发病株率", "灰斑病病情指数",
            "大斑病发病株率", "大斑病病情指数",
            "白斑病发病株率", "白斑病病情指数",
        ]

    def _expand_daily_with_interpolation(
        self,
        merged_df: pd.DataFrame,
        target_cols: list[str],
        method: str = "linear",
    ) -> pd.DataFrame:
        print("正在按日补全调查值（插值）...")

        if self.weather_data is None:
            raise ValueError("请先加载气象数据")

        merged_df = merged_df.copy()
        merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce").dt.normalize()

        daily_frames = []
        for group_key, gdf in merged_df.groupby("group_key"):
            gdf = gdf.sort_values("date").copy()
            if gdf["date"].isna().all():
                continue

            base_site = gdf["地点"].iloc[0]
            base_variety = gdf["品种"].iloc[0] if "品种" in gdf.columns else "未知品种"
            matched_location = gdf["匹配气象地点"].iloc[0] if "匹配气象地点" in gdf.columns else base_site

            weather_df = self.weather_data[self.weather_data["地点"] == matched_location].copy()
            if weather_df.empty:
                continue

            weather_df = weather_df.sort_values("date").copy()
            weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce").dt.normalize()
            min_date = gdf["date"].min()
            max_date = gdf["date"].max()
            weather_df = weather_df[(weather_df["date"] >= min_date) & (weather_df["date"] <= max_date)].copy()
            if weather_df.empty:
                continue

            daily = weather_df.copy()
            daily["地点"] = base_site
            daily["品种"] = base_variety
            daily["group_key"] = group_key
            daily["匹配气象地点"] = matched_location

            disease_cols = [c for c in target_cols if c in gdf.columns]
            if "growth_stage_code" in gdf.columns:
                disease_cols.append("growth_stage_code")

            disease_vals = gdf[["date"] + disease_cols].drop_duplicates(subset=["date"]).copy()
            daily = daily.merge(disease_vals, on="date", how="left")

            daily["is_survey"] = daily["date"].isin(gdf["date"]).astype(int)

            for col in disease_cols:
                if col in target_cols:
                    daily[col] = pd.to_numeric(daily[col], errors="coerce")
                    daily[col] = daily[col].interpolate(method=method, limit_direction="both")
                elif col == "growth_stage_code":
                    daily[col] = pd.to_numeric(daily[col], errors="coerce")
                    daily[col] = daily[col].ffill().bfill()

            daily_frames.append(daily)

        if not daily_frames:
            raise ValueError("日尺度补全失败：未生成任何可用样本")

        out_df = pd.concat(daily_frames, ignore_index=True)
        out_df = out_df.sort_values(["地点", "品种", "date"]).reset_index(drop=True)
        print(f"日尺度补全完成，共 {len(out_df)} 条样本")
        return out_df

    def merge_data_by_location(self, interpolate_daily: bool = False, interpolate_method: str = "linear") -> pd.DataFrame:
        print("正在按真实调查日期匹配病害与气象特征...")

        if self.disease_data is None or self.weather_data is None:
            raise ValueError("请先加载病害数据和气象数据")

        target_cols = self.get_target_columns()

        df_disease = self.disease_data.copy().replace("-", np.nan)

        if "地点" not in df_disease.columns:
            raise ValueError("病害数据缺少 地点 列")

        if "时间" not in df_disease.columns:
            raise ValueError("病害数据缺少 时间 列")

        for col in target_cols:
            if col in df_disease.columns:
                df_disease[col] = self._to_numeric_keep_nan(df_disease[col])

        if "品种" not in df_disease.columns:
            df_disease["品种"] = "未知品种"

        df_disease["品种"] = (
            df_disease["品种"]
            .replace(["", " ", "-", "无", "None", "nan"], np.nan)
            .astype("string")
            .str.strip()
            .fillna("未知品种")
        )

        df_disease["时间"] = df_disease["时间"].apply(self._parse_any_date)
        df_disease = df_disease[df_disease["时间"].notna()].copy()

        df_disease = df_disease.sort_values(["地点", "品种", "时间"]) 

        # 同一天同地点同品种可能有多条调查记录，先聚合
        agg_map = {}

        for col in df_disease.columns:
            if col in ["地点", "品种", "时间"]:
                continue
            elif col in target_cols:
                agg_map[col] = "mean"
            else:
                agg_map[col] = "first"

        df_disease = (
            df_disease
            .groupby(["地点", "品种", "时间"], as_index=False)
            .agg(agg_map)
        )

        def _build_group_key(row: pd.Series) -> str:
            site = str(row.get("地点", "")).strip()
            variety = str(row.get("品种", "")).strip()
            if variety in {"", "未知品种", "nan", "None"}:
                return site
            return f"{site}_{variety}"

        df_disease["group_key"] = df_disease.apply(_build_group_key, axis=1)

        # 生成病害增量目标
        for col in target_cols:
            if col in df_disease.columns:
                df_disease[f"{col}_prev"] = (
                    df_disease
                    .groupby(["地点", "品种"])[col]
                    .shift(1)
                )

                df_disease[f"{col}_delta"] = df_disease[col] - df_disease[f"{col}_prev"]

                df_disease[f"{col}_delta"] = df_disease[f"{col}_delta"].clip(lower=0)

        available_locations = list(self.weather_data["地点"].unique())
        merged_list = []

        # 尽量按原始调查日期精确匹配气象数据，避免对目标插值
        for _, disease_row in df_disease.iterrows():
            location = disease_row["地点"]
            survey_date = pd.to_datetime(disease_row["时间"], errors="coerce")

            if pd.isna(survey_date):
                continue

            survey_date = survey_date.normalize()
            matched_location = self._fuzzy_match_location(location, available_locations)

            weather_subset = self.weather_data[
                (self.weather_data["地点"] == matched_location) &
                (self.weather_data["date"] == survey_date)
            ]

            if weather_subset.empty:
                # 如果当天没有匹配的气象记录，跳过该调查（避免插值）
                continue

            w = weather_subset.iloc[0].to_dict()
            d = disease_row.to_dict()

            # 生育期编码：尝试在病害表中查找含有 生育 字样的列
            growth_stage = None
            for c in df_disease.columns:
                if "育" in c:
                    growth_stage = d.get(c)
                    break

            d["growth_stage_code"] = self._encode_growth_stage(growth_stage)

            # 将气象前缀列合并
            merged = {
                "地点": location,
                "品种": d.get("品种", "未知品种"),
                "date": survey_date,
                "group_key": d.get("group_key"),
                "匹配气象地点": matched_location,
            }

            # 添加病害相关列
            for k, v in d.items():
                if k in ["地点", "时间"]:
                    continue
                merged[k] = v

            # 添加气象特征（保留原始列名）
            for k, v in w.items():
                if k in ["地点", "date"]:
                    continue
                merged[k] = v

            merged_list.append(merged)

        if not merged_list:
            raise ValueError("未能匹配到任何病害与气象记录，请检查数据和地点名称匹配")

        merged_df = pd.DataFrame(merged_list)
        # 规范列名顺序
        merged_df = merged_df.sort_values(["地点", "品种", "date"]).reset_index(drop=True)

        if interpolate_daily:
            merged_df = self._expand_daily_with_interpolation(
                merged_df,
                target_cols=target_cols,
                method=interpolate_method,
            )

        self.merged_data = merged_df
        print(f"合并完成，共 {len(self.merged_data)} 条样本")
        return self.merged_data

    def get_feature_columns(self) -> list[str]:
        if self.merged_data is None:
            return []

        exclude = set(self.get_target_columns())
        exclude.update({"地点", "品种", "date"})
        # 排除 prev/delta 列
        exclude.update([c for c in self.merged_data.columns if str(c).endswith("_prev") or str(c).endswith("_delta")])

        feature_cols = [
            c for c in self.merged_data.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(self.merged_data[c])
        ]

        return feature_cols

    def get_xgb_train_data(
        self,
        target_col: str = "灰斑病发病株率_delta",
        drop_na_target: bool = True
    ):
        if self.merged_data is None:
            raise ValueError("请先执行数据匹配")

        if target_col not in self.merged_data.columns:
            raise ValueError(f"目标列不存在: {target_col}")

        feature_cols = self.get_feature_columns()

        df = self.merged_data.copy()

        if drop_na_target:
            df = df[df[target_col].notna()].copy()

        X = df[feature_cols]
        y = df[target_col]

        return X, y, feature_cols

    # =========================
    # 保存数据
    # =========================

    def save_processed_data(self, output_file: str = "data/processed_data_xgb.csv") -> None:
        if self.merged_data is None:
            raise ValueError("请先执行数据匹配")

        out_dir = os.path.dirname(output_file)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        self.merged_data.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"处理后的数据已保存至: {output_file}")

    def save_feature_columns(self, output_file: str = "data/xgb_feature_columns.txt") -> None:
        feature_cols = self.get_feature_columns()

        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for col in feature_cols:
                f.write(col + "\n")

        print(f"XGBoost 特征列已保存至: {output_file}")

    # =========================
    # 总流程
    # =========================

    def process_all(
        self,
        output_file: str = "data/processed_data_xgb.csv",
        feature_file: str = "data/xgb_feature_columns.txt",
        interpolate_daily: bool = False,
        interpolate_method: str = "linear",
    ) -> pd.DataFrame:
        print("=" * 60)
        print("开始数据预处理流程（XGBoost 优化版）")
        print("=" * 60)

        self.load_disease_data()
        self.load_weather_data()
        self.generate_derived_features()
        self.merge_data_by_location(
            interpolate_daily=interpolate_daily,
            interpolate_method=interpolate_method,
        )
        self.save_processed_data(output_file)
        self.save_feature_columns(feature_file)

        print("=" * 60)
        print("数据预处理完成！")
        print("=" * 60)

        return self.merged_data


if __name__ == "__main__":
    disease_file = "data/2025定点监测叶斑病调查数据.xlsx"
    weather_file = "data/2025年定点监测气象数据.xlsx"

    preprocessor = CornDiseaseDataPreprocessor(
        disease_file=disease_file,
        weather_file=weather_file
    )

    processed_data = preprocessor.process_all(
        output_file="data/processed_data_xgb.csv",
        feature_file="data/xgb_feature_columns.txt"
    )

    print("\n处理后数据预览:")
    print(processed_data.head())

    print(f"\n数据形状: {processed_data.shape}")
    print(f"特征列数量: {len(preprocessor.get_feature_columns())}")

    print("\n推荐训练目标列:")
    for col in processed_data.columns:
        if col.endswith("_delta"):
            print(" -", col)