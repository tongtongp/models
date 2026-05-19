"""
玉米病害预测模块
功能：
1. 加载训练好的模型
2. 根据气象数据预测病害发病率
3. 批量预测和结果输出
"""

import pandas as pd
import numpy as np
import joblib
import os
import re
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from data_preprocessing import CornDiseaseDataPreprocessor
import warnings
warnings.filterwarnings('ignore')

def _silent_print(*args, **kwargs):
    return None


print = _silent_print

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class CornDiseaseForecast:
    """玉米病害预测类"""
    
    def __init__(self, models_dir='models', output_root='results'):
        """
        初始化预测器
        
        Args:
            models_dir: 模型文件所在目录
        """
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.imputers = {}
        self.feature_columns = None
        self.feature_columns_by_model = {}
        self.output_root = output_root
        self.output_dirs = None
        
        # 三种病害
        self.diseases = ['灰斑病', '大斑病', '白斑病']
        self.target_types = ['rate', 'index']
        self.actual_rate_cols = {
            '灰斑病': '灰斑病发病株率',
            '大斑病': '大斑病发病株率',
            '白斑病': '白斑病发病株率'
        }

    def _prepare_output_dirs(self, run_tag=None):
        """创建更清晰的输出目录结构：表格与图片分开"""
        if run_tag is None:
            run_tag = datetime.now().strftime('run_%Y%m%d_%H%M%S')

        run_root = os.path.join(self.output_root, run_tag)
        output_dirs = {
            'root': run_root,
            'tables_root': os.path.join(run_root, 'tables'),
            'figures_root': os.path.join(run_root, 'figures'),
            'daily_tables': os.path.join(run_root, 'tables', 'daily_predictions'),
            'weekly_tables': os.path.join(run_root, 'tables', 'weekly_predictions'),
            'compare_tables': os.path.join(run_root, 'tables', 'prediction_vs_actual'),
            'reports': os.path.join(run_root, 'tables', 'reports'),
            'daily_figures': os.path.join(run_root, 'figures', 'daily_predictions'),
            'compare_figures': os.path.join(run_root, 'figures', 'prediction_vs_actual'),
            'overview_figures': os.path.join(run_root, 'figures', 'overview')
        }

        for p in output_dirs.values():
            os.makedirs(p, exist_ok=True)

        self.output_dirs = output_dirs
        print(f"\n📁 输出目录: {run_root}")
        return output_dirs

    def _compact_daily_table(self, daily_predictions):
        """输出更简洁的日尺度预测表"""
        df = daily_predictions.copy()
        disease_cols = [f'{d}_发病概率(%)' for d in self.diseases if f'{d}_发病概率(%)' in df.columns]
        risk_cols = [f'{d}_风险等级' for d in self.diseases if f'{d}_风险等级' in df.columns]

        if disease_cols:
            df['综合发病概率(%)'] = df[disease_cols].mean(axis=1).round(2)
            df['主要风险病害'] = df[disease_cols].idxmax(axis=1).str.replace('_发病概率\(%\)', '', regex=True)
            if '综合风险等级' not in df.columns:
                df['综合风险等级'] = self._probability_to_risk_level(df['综合发病概率(%)'])

        keep_cols = ['date', '地点'] + disease_cols + risk_cols + ['综合发病概率(%)', '综合风险等级', '主要风险病害']
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols]

        for c in disease_cols + ['综合发病概率(%)']:
            if c in df.columns:
                df[c] = df[c].round(2)

        return df

    def _compact_weekly_table(self, weekly_predictions):
        """输出更简洁的周预测表"""
        df = weekly_predictions.copy()
        disease_cols = [f'{d}_发病概率(%)' for d in self.diseases if f'{d}_发病概率(%)' in df.columns]
        risk_cols = [f'{d}_风险等级' for d in self.diseases if f'{d}_风险等级' in df.columns]

        if disease_cols:
            df['综合发病概率(%)'] = df[disease_cols].mean(axis=1).round(2)
            top_disease = df[disease_cols].idxmax(axis=1).str.replace('_发病概率\(%\)', '', regex=True)
            df['主要风险病害'] = top_disease
            if '综合风险等级' not in df.columns:
                df['综合风险等级'] = self._probability_to_risk_level(df['综合发病概率(%)'])

        keep_cols = ['date', '地点'] + disease_cols + risk_cols + ['综合发病概率(%)', '综合风险等级', '主要风险病害']
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols]

        for c in disease_cols + ['综合发病概率(%)']:
            if c in df.columns:
                df[c] = df[c].round(2)

        return df

    def _compact_comparison_table(self, comparison_df):
        """输出更简洁的预测-真实值对比表"""
        if comparison_df is None or len(comparison_df) == 0:
            return comparison_df

        df = comparison_df.copy()
        keep_cols = ['date', '地点']

        for disease in self.diseases:
            pred_col = f'{disease}_发病概率(%)'
            true_col = f'{disease}_真实发病株率(%)'
            err_col = f'{disease}_绝对误差'
            for col in [pred_col, true_col, err_col]:
                if col in df.columns:
                    keep_cols.append(col)

        df = df[keep_cols]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(2)
        return df

    def _sanitize_filename(self, name):
        """将地点名转换为安全文件名"""
        safe = re.sub(r'[\\/:*?"<>|\s]+', '_', str(name).strip())
        return safe.strip('._') or 'unknown_location'

    def _get_effective_weather_start_date(self, location, requested_start_date, actual_file='data/processed_data.csv', lookback_days=7):
        """
        根据真实调查数据，计算气象数据有效起始日期：首次调查日前 lookback_days 天。

        若真实值不可用，则回退为 requested_start_date。
        """
        requested_start_dt = pd.to_datetime(requested_start_date)

        if not os.path.exists(actual_file):
            return requested_start_dt

        try:
            actual_df = pd.read_csv(actual_file, encoding='utf-8-sig')
        except Exception:
            return requested_start_dt

        if 'date' not in actual_df.columns:
            return requested_start_dt

        actual_df['date'] = pd.to_datetime(actual_df['date'], errors='coerce')
        actual_df = actual_df.dropna(subset=['date'])

        if location is not None and '地点' in actual_df.columns:
            loc_mask = actual_df['地点'].astype(str).str.strip() == str(location).strip()
            actual_df = actual_df[loc_mask]

        if len(actual_df) == 0:
            return requested_start_dt

        first_survey_date = actual_df['date'].min().normalize()
        weather_start_by_survey = first_survey_date - pd.Timedelta(days=lookback_days)

        # 不早于用户请求的开始日期
        effective_start = max(requested_start_dt.normalize(), weather_start_by_survey)
        return effective_start

    def _to_weekly_points(self, predictions, origin_date=None):
        """将日尺度预测压缩为周尺度（一周一个点位）"""
        weekly = predictions.copy()
        weekly['date'] = pd.to_datetime(weekly['date'])
        weekly = weekly.sort_values('date').drop_duplicates(subset=['date'])

        numeric_cols = weekly.select_dtypes(include=[np.number]).columns.tolist()
        resample_kwargs = {'rule': '7D'}
        if origin_date is not None:
            resample_kwargs['origin'] = pd.to_datetime(origin_date)
        weekly_agg = weekly.set_index('date')[numeric_cols].resample(**resample_kwargs).mean().reset_index()

        # 生成完整的周期日期范围，填充缺失的周
        if len(weekly_agg) > 0:
            start_date = weekly_agg['date'].min()
            end_date = weekly_agg['date'].max()
            # 创建完整的周期日期序列
            date_range = pd.date_range(start=start_date, end=end_date, freq='7D')
            full_dates_df = pd.DataFrame({'date': date_range})
            # 与已有数据合并，缺失的周使用线性插值填充
            weekly_agg = full_dates_df.merge(weekly_agg, on='date', how='left')
            # 对数值列进行线性插值
            for col in numeric_cols:
                if col in weekly_agg.columns:
                    weekly_agg[col] = weekly_agg[col].interpolate(method='linear')

        if '地点' in predictions.columns and len(predictions) > 0:
            weekly_agg['地点'] = predictions['地点'].iloc[0]

        # 保持列顺序
        ordered_cols = ['date', '地点'] + [c for c in weekly_agg.columns if c not in ['date', '地点']]
        ordered_cols = [c for c in ordered_cols if c in weekly_agg.columns]
        return weekly_agg[ordered_cols]

    def get_actual_weekly_rates(self, location, start_date, end_date, actual_file='data/processed_data.csv'):
        """读取真实发病株率并按周聚合，用于与预测概率对比"""
        if not os.path.exists(actual_file):
            print(f"⚠️ 未找到真实值文件: {actual_file}")
            return None

        actual_df = pd.read_csv(actual_file, encoding='utf-8-sig')
        if 'date' not in actual_df.columns or '地点' not in actual_df.columns:
            print("⚠️ 真实值文件缺少 date 或 地点 列，跳过真实值对比")
            return None

        actual_df['date'] = pd.to_datetime(actual_df['date'], errors='coerce')
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        subset = actual_df[
            (actual_df['地点'].astype(str).str.strip() == str(location).strip()) &
            (actual_df['date'] >= start_dt) &
            (actual_df['date'] <= end_dt)
        ].copy()

        if len(subset) == 0:
            print(f"⚠️ 地点 {location} 在真实值文件中无可对齐样本")
            return None

        existing_cols = [c for c in self.actual_rate_cols.values() if c in subset.columns]
        if not existing_cols:
            print("⚠️ 真实值文件缺少发病株率列，跳过真实值对比")
            return None

        # 同一天可能有多条记录，先按天平均
        daily_actual = subset.groupby('date')[existing_cols].mean().reset_index()

        # 与预测保持同一周起点
        weekly_actual = daily_actual.set_index('date')[existing_cols].resample('7D', origin=start_dt).mean().reset_index()

        # 生成完整的周期日期范围，填充缺失的周
        if len(weekly_actual) > 0:
            actual_start_date = weekly_actual['date'].min()
            actual_end_date = weekly_actual['date'].max()
            # 创建完整的周期日期序列
            date_range = pd.date_range(start=actual_start_date, end=actual_end_date, freq='7D')
            full_dates_df = pd.DataFrame({'date': date_range})
            # 与已有数据合并，缺失的周使用线性插值填充
            weekly_actual = full_dates_df.merge(weekly_actual, on='date', how='left')
            # 对数值列进行线性插值
            for col in existing_cols:
                if col in weekly_actual.columns:
                    weekly_actual[col] = weekly_actual[col].interpolate(method='linear')

        rename_map = {
            self.actual_rate_cols[d]: f'{d}_真实发病株率(%)'
            for d in self.diseases
            if self.actual_rate_cols[d] in weekly_actual.columns
        }
        weekly_actual = weekly_actual.rename(columns=rename_map)
        weekly_actual['地点'] = location
        return weekly_actual

    def build_prediction_actual_comparison(self, weekly_predictions, location, start_date, end_date, actual_file='data/processed_data.csv'):
        """构建预测概率与真实发病株率对比表"""
        weekly_actual = self.get_actual_weekly_rates(location, start_date, end_date, actual_file=actual_file)
        if weekly_actual is None:
            return None

        comp = weekly_predictions.copy()
        comp['date'] = pd.to_datetime(comp['date'])
        weekly_actual['date'] = pd.to_datetime(weekly_actual['date'])
        comp = comp.merge(weekly_actual, on=['date', '地点'], how='left')

        # 仅保留“有真实发病株率”的时间之后的数据，确保预测图从真实值起点开始
        actual_cols = [f'{d}_真实发病株率(%)' for d in self.diseases if f'{d}_真实发病株率(%)' in comp.columns]
        if actual_cols:
            valid_actual_mask = comp[actual_cols].notna().any(axis=1)
            if valid_actual_mask.any():
                first_valid_date = comp.loc[valid_actual_mask, 'date'].min()
                comp = comp[comp['date'] >= first_valid_date].copy()
            else:
                # 没有任何可对齐真实值时，返回空并由上层跳过绘图
                return None

        # 误差列：预测概率 - 真实发病株率
        for disease in self.diseases:
            pred_col = f'{disease}_发病概率(%)'
            true_col = f'{disease}_真实发病株率(%)'
            if pred_col in comp.columns and true_col in comp.columns:
                comp[f'{disease}_绝对误差'] = (comp[pred_col] - comp[true_col]).abs()

        return comp

    def plot_prediction_vs_actual(self, comparison_df, location, output_file):
        """绘制预测概率与真实值对比图（周尺度）"""
        if comparison_df is None or len(comparison_df) == 0:
            return

        df = comparison_df.copy().sort_values('date')
        df['date'] = pd.to_datetime(df['date'])

        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for i, disease in enumerate(self.diseases):
            ax = axes[i]
            pred_col = f'{disease}_发病概率(%)'
            true_col = f'{disease}_真实发病株率(%)'

            # 分别删除各列的NaN值，确保线条连续
            if pred_col in df.columns:
                pred_data = df[['date', pred_col]].dropna()
                ax.plot(pred_data['date'], pred_data[pred_col], marker='o', markersize=4, linewidth=1.8, label='预测', drawstyle='default')
            if true_col in df.columns and df[true_col].notna().any():
                true_data = df[['date', true_col]].dropna()
                ax.plot(true_data['date'], true_data[true_col], marker='s', markersize=4, linewidth=1.8, label='真实', drawstyle='default')

            ax.set_title(f'{location} - {disease}')
            ax.set_ylabel('%')
            ax.set_ylim(-5, 105)
            ax.grid(True, alpha=0.2, linestyle='--')
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='upper right', frameon=False)

        axes[-1].set_xlabel('日期')
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_daily_prediction(self, daily_df, location, output_file):
        """绘制日尺度三病害预测曲线图"""
        if daily_df is None or len(daily_df) == 0:
            return

        df = daily_df.copy().sort_values('date')
        df['date'] = pd.to_datetime(df['date'])
        
        # 删除值为NaN的行，确保连续线条
        df = df.dropna(subset=[c for c in df.columns if c.endswith('发病概率(%)')])

        fig, ax = plt.subplots(figsize=(11, 4.5))
        for disease in self.diseases:
            col = f'{disease}_发病概率(%)'
            if col in df.columns:
                ax.plot(df['date'], df[col], marker='o', markersize=3, linewidth=1.6, label=disease, drawstyle='default')

        ax.set_title(f'{location} - 日尺度病害预测')
        ax.set_ylabel('%')
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.legend(loc='upper right', frameon=False, ncol=3)
        ax.set_xlabel('日期')
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_weekly_prediction(self, weekly_df, location, output_file):
        """绘制周尺度三病害预测曲线图（一周一个点）"""
        if weekly_df is None or len(weekly_df) == 0:
            return

        df = weekly_df.copy().sort_values('date')
        df['date'] = pd.to_datetime(df['date'])
        
        # 删除值为NaN的行，确保连续线条
        df = df.dropna(subset=[c for c in df.columns if c.endswith('发病概率(%)')])

        fig, ax = plt.subplots(figsize=(11, 4.5))
        for disease in self.diseases:
            col = f'{disease}_发病概率(%)'
            if col in df.columns:
                ax.plot(df['date'], df[col], marker='o', markersize=4, linewidth=1.8, label=disease, drawstyle='default')

        ax.set_title(f'{location} - 周尺度病害预测（每周1点）')
        ax.set_ylabel('%')
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.legend(loc='upper right', frameon=False, ncol=3)
        ax.set_xlabel('日期')
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def export_weekly_from_predictions(self, predictions, output_dir='outputs', figures_dir=None, tables_dir=None):
        """将预测结果按地点压缩为周尺度并导出CSV和图（支持图表/表格分目录）"""
        if predictions is None or len(predictions) == 0:
            return None

        if figures_dir is None or tables_dir is None:
            # 向后兼容：未单独传入时仍写入 output_dir
            figures_dir = output_dir
            tables_dir = output_dir

        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)
        pred = predictions.copy()
        pred['date'] = pd.to_datetime(pred['date'])

        all_weekly = []
        for location, group in pred.groupby('地点'):
            weekly = self._to_weekly_points(group)
            weekly_compact = self._compact_weekly_table(weekly)
            all_weekly.append(weekly_compact)

            safe_loc = self._sanitize_filename(location)
            weekly_plot = os.path.join(figures_dir, f'weekly_prediction_{safe_loc}.png')
            weekly_csv = os.path.join(tables_dir, f'weekly_prediction_{safe_loc}.csv')
            weekly_compact.to_csv(weekly_csv, index=False, encoding='utf-8-sig')
            self.plot_weekly_prediction(weekly_compact, location, weekly_plot)
            print(f"✓ 周尺度预测图已保存: {weekly_plot}")

        merged_weekly = pd.concat(all_weekly, ignore_index=True) if all_weekly else None
        if merged_weekly is not None and len(merged_weekly) > 0:
            merged_csv = os.path.join(tables_dir, 'predictions_all_locations_weekly.csv')
            merged_weekly.to_csv(merged_csv, index=False, encoding='utf-8-sig')
            print(f"✓ 周尺度汇总表已保存: {merged_csv}")

        return merged_weekly

    def plot_disease_overview_all_locations(self, all_comparison_results, output_dir='outputs'):
        """按病害输出所有地点总览大图（预测概率 vs 真实发病株率）"""
        if not all_comparison_results:
            print("⚠️ 无可用的预测-真实值对比数据，跳过总览图")
            return

        os.makedirs(output_dir, exist_ok=True)
        locations = list(all_comparison_results.keys())
        n_locs = len(locations)
        n_cols = 3 if n_locs > 4 else 2
        n_rows = math.ceil(n_locs / n_cols)

        for disease in self.diseases:
            pred_col = f'{disease}_发病概率(%)'
            true_col = f'{disease}_真实发病株率(%)'

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 3.8 * n_rows), sharex=False, sharey=True)
            axes = axes.flatten()

            for i, loc in enumerate(locations):
                ax = axes[i]
                df = all_comparison_results[loc].copy()
                if df is None or len(df) == 0:
                    ax.set_title(f'{loc}\n无数据')
                    ax.axis('off')
                    continue

                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')

                # 分别删除各列的NaN值，确保线条连续
                if pred_col in df.columns:
                    pred_data = df[['date', pred_col]].dropna()
                    ax.plot(pred_data['date'], pred_data[pred_col], marker='o', markersize=3.5, linewidth=1.6, label='预测', drawstyle='default')
                if true_col in df.columns and df[true_col].notna().any():
                    true_data = df[['date', true_col]].dropna()
                    ax.plot(true_data['date'], true_data[true_col], marker='s', markersize=3.5, linewidth=1.6, label='真实', drawstyle='default')

                ax.set_title(loc)
                ax.set_ylim(-5, 105)
                ax.grid(True, alpha=0.2, linestyle='--')

                # 仅首个子图显示图例，避免遮挡
                if i == 0:
                    ax.legend(loc='upper right', frameon=False)

            # 隐藏多余子图
            for j in range(len(locations), len(axes)):
                axes[j].axis('off')

            fig.suptitle(f'{disease} - {n_locs}个地点周尺度预测与真实值总览', fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.97])

            out_file = os.path.join(output_dir, f'overview_{n_locs}sites_{disease}_预测_vs_真实.png')
            fig.savefig(out_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ 总览图已保存: {out_file}")

    def batch_predict_all_locations_weekly(self, weather_file, start_date, end_date, locations=None, actual_file='data/processed_data.csv', output_root=None):
        """
        批量预测多个地点并输出周尺度图像（一周一个点位）

        Returns:
            dict: 每个地点对应的周尺度预测DataFrame
        """
        if output_root is not None:
            self.output_root = output_root
        out_dirs = self._prepare_output_dirs()

        # 自动读取地点（默认全部有效气象sheet）
        if locations is None:
            excel_file = pd.ExcelFile(weather_file)
            locations = [
                sn for sn in excel_file.sheet_names
                if ('站点' not in sn and '说明' not in sn and '统计' not in sn)
            ]

        all_weekly_results = {}
        all_comparison_results = {}

        print(f"\n开始批量预测，共 {len(locations)} 个地点...")
        for i, location in enumerate(locations, 1):
            print(f"\n[{i}/{len(locations)}] 正在预测地点: {location}")
            try:
                # 先做日尺度预测
                daily_predictions = self.predict_from_weather_file(
                    weather_file=weather_file,
                    location=location,
                    start_date=start_date,
                    end_date=end_date,
                    output_file=None,
                    actual_file=actual_file,
                    lookback_days=7
                )

                # 输出日尺度CSV和图
                safe_loc = self._sanitize_filename(location)
                compact_daily = self._compact_daily_table(daily_predictions)
                daily_csv = os.path.join(out_dirs['daily_tables'], f"{safe_loc}_daily_prediction.csv")
                daily_plot = os.path.join(out_dirs['daily_figures'], f"{safe_loc}_daily_prediction.png")
                compact_daily.to_csv(daily_csv, index=False, encoding='utf-8-sig')
                self.plot_daily_prediction(compact_daily, location, daily_plot)
                print(f"✓ 日尺度预测表已保存: {daily_csv}")
                print(f"✓ 日尺度预测图已保存: {daily_plot}")

                # 压缩到周尺度（每周一个点位）
                weekly_predictions = self._to_weekly_points(daily_predictions, origin_date=start_date)
                compact_weekly = self._compact_weekly_table(weekly_predictions)
                all_weekly_results[location] = compact_weekly

                # 输出周尺度CSV
                weekly_csv = os.path.join(out_dirs['weekly_tables'], f"{safe_loc}_weekly_prediction.csv")
                compact_weekly.to_csv(weekly_csv, index=False, encoding='utf-8-sig')

                # 输出真实值对比（若可获取）
                comp_df = self.build_prediction_actual_comparison(
                    weekly_predictions,
                    location,
                    start_date,
                    end_date,
                    actual_file=actual_file
                )
                if comp_df is not None:
                    compact_comp = self._compact_comparison_table(comp_df)
                    comp_csv = os.path.join(out_dirs['compare_tables'], f"{safe_loc}_prediction_vs_actual.csv")
                    comp_plot = os.path.join(out_dirs['compare_figures'], f"{safe_loc}_prediction_vs_actual.png")
                    compact_comp.to_csv(comp_csv, index=False, encoding='utf-8-sig')
                    self.plot_prediction_vs_actual(comp_df, location, comp_plot)
                    all_comparison_results[location] = compact_comp
                    print(f"✓ 预测-真实值对比表已保存: {comp_csv}")
                    print(f"✓ 预测-真实值对比图已保存: {comp_plot}")

                print(f"✓ 周尺度预测已保存: {weekly_csv}")

            except Exception as e:
                print(f"✗ 地点 {location} 预测失败: {str(e)}")

        print("\n批量预测完成")
        return all_weekly_results, all_comparison_results

    def _calculate_disease_probability(self, rate_predictions, index_predictions):
        """
        由发病株率与病情指数综合计算发病概率（0~100%）

        说明：
        - 发病株率天然是百分比量纲；
        - 病情指数常见范围约 0~50，这里线性映射到 0~100 后参与融合；
        - 最终采用加权融合并裁剪到 0~100。
        """
        rate_component = np.clip(np.asarray(rate_predictions), 0, 100)
        index_component = np.clip(np.asarray(index_predictions) * 2.0, 0, 100)
        probability = rate_component 
        # probability = 0.7 * rate_component + 0.3 * index_component
        return np.clip(probability, 0, 100)

    def _probability_to_risk_level(self, probability_values):
        """将发病概率映射为低/中/高风险等级。"""
        p = np.clip(np.asarray(probability_values, dtype=float), 0, 100)
        levels = np.select(
            [p < 30, p < 70],
            ['低风险', '中风险'],
            default='高风险'
        )
        if isinstance(probability_values, pd.Series):
            return pd.Series(levels, index=probability_values.index)
        return levels
    
    def load_models(self):
        """加载所有训练好的模型和标准化器"""
        print("正在加载模型...")

        # 优先加载 full bundle（每个病害一个 .pt，包含 rate/index 两个目标）
        for disease in self.diseases:
            bundle_file = f"{self.models_dir}/full_bundle_{disease}.pt"
            if not os.path.exists(bundle_file):
                continue
            try:
                bundle = joblib.load(bundle_file)
                models = bundle.get('models', {})
                scalers = bundle.get('scalers', {})
                imputers = bundle.get('imputers', {})
                feature_columns = bundle.get('feature_columns', {})

                ok_count = 0
                for target_type in self.target_types:
                    model_key = f"{disease}_{target_type}"
                    if target_type in models and target_type in scalers and target_type in imputers:
                        self.models[model_key] = models[target_type]
                        self.scalers[model_key] = scalers[target_type]
                        self.imputers[model_key] = imputers[target_type]
                        if target_type in feature_columns:
                            self.feature_columns_by_model[model_key] = list(feature_columns[target_type])
                        ok_count += 1

                print(f"[成功] {disease} full bundle 加载成功 ({os.path.basename(bundle_file)}), 子模型数={ok_count}")
            except Exception as e:
                print(f"[失败] {disease} full bundle 加载失败: {str(e)}")
        
        for disease in self.diseases:
            for target_type in self.target_types:
                model_key = f"{disease}_{target_type}"
                if model_key in self.models:
                    continue
                model_file_pt = f"{self.models_dir}/model_{disease}_{target_type}.pt"
                model_file_pkl = f"{self.models_dir}/model_{disease}_{target_type}.pkl"
                scaler_file = f"{self.models_dir}/scaler_{disease}_{target_type}.pkl"
                imputer_file = f"{self.models_dir}/imputer_{disease}_{target_type}.pkl"
                
                try:
                    # 优先加载 .pt，若不存在则回退到 .pkl
                    model_file = model_file_pt if os.path.exists(model_file_pt) else model_file_pkl
                    self.models[model_key] = joblib.load(model_file)
                    self.scalers[model_key] = joblib.load(scaler_file)
                    self.imputers[model_key] = joblib.load(imputer_file)
                    print(f"[成功] {model_key} 模型加载成功 ({os.path.basename(model_file)})")
                except FileNotFoundError:
                    print(f"[失败] {model_key} 模型文件未找到")
                except Exception as e:
                    print(f"[失败] {model_key} 模型加载失败: {str(e)}")
        
        print(f"\n共加载 {len(self.models)} 个模型")
    
    def prepare_weather_features(self, weather_file, location, start_date, end_date, actual_file='data/processed_data.csv', lookback_days=7):
        """
        准备气象特征数据
        
        Args:
            weather_file: 气象数据文件
            location: 地点名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            处理后的特征数据
        """
        print(f"\n正在准备气象特征数据...")
        if location is None:
            print("地点: 自动使用气象文件中的全部有效地点")
        else:
            print(f"地点: {location}")
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        effective_start_date = self._get_effective_weather_start_date(
            location=location,
            requested_start_date=start_date,
            actual_file=actual_file,
            lookback_days=lookback_days
        )

        print(f"时间范围(用户输入): {start_date.date()} 至 {end_date.date()}")
        print(f"时间范围(实际用于气象): {effective_start_date.date()} 至 {end_date.date()}")
        
        # 读取气象数据
        excel_file = pd.ExcelFile(weather_file)
        
        # 查找对应地点的sheet，支持模糊匹配
        sheet_names = excel_file.sheet_names
        matched_sheet = None

        # 不指定地点时：直接使用全部有效气象sheet，增强普适性
        if location is None:
            valid_sheets = [
                sn for sn in sheet_names
                if ('站点' not in sn and '说明' not in sn and '统计' not in sn)
            ]
            all_weather_data = []
            for sn in valid_sheets:
                df = pd.read_excel(weather_file, sheet_name=sn)
                date_col = 'date' if 'date' in df.columns else ('日期' if '日期' in df.columns else None)
                if date_col:
                    def parse_date(x):
                        s = str(x).strip().split('.')[0]
                        if len(s) >= 8 and s[:8].isdigit():
                            try:
                                return pd.to_datetime(s[:8], format='%Y%m%d')
                            except ValueError:
                                pass
                        return pd.to_datetime(x)

                    df['date'] = df[date_col].apply(parse_date).dt.normalize()
                else:
                    continue

                df = df[(df['date'] >= effective_start_date) & (df['date'] <= end_date)].copy()
                if len(df) == 0:
                    continue

                df['地点'] = sn
                all_weather_data.append(df)

            if not all_weather_data:
                raise ValueError("指定日期范围内没有可用的气象数据")

            weather_data = pd.concat(all_weather_data, ignore_index=True)

            preprocessor = CornDiseaseDataPreprocessor(None, None)
            preprocessor.weather_data = weather_data
            processed_data = preprocessor.generate_derived_features()
            processed_data["month"] = pd.to_datetime(processed_data["date"]).dt.month.astype(int)
            processed_data["day_of_year"] = pd.to_datetime(processed_data["date"]).dt.dayofyear.astype(int)

            print(f"✓ 气象数据处理完成，共 {len(processed_data)} 条记录（全部地点）")
            return processed_data
        
        if location in sheet_names:
            matched_sheet = location
        else:
            location_str = str(location).strip()
            # 引入更强的模糊匹配，复用预处理阶段清洗核心名字的逻辑
            import re
            def get_core_name(name):
                s = name
                s = re.sub(r'^\d+[\.、\s]*', '', s)
                for prefix in ['四川省', '四川', '云南省', '云南', '雅安市', '雅安', '成都市', '成都', '甘孜州', '甘孜', '德宏州', '德宏']:
                    if s.startswith(prefix):
                        s = s[len(prefix):]
                return s
            
            core_loc = get_core_name(location_str)
            for sn in sheet_names:
                core_sn = get_core_name(sn)
                if len(core_loc) >= 2 and len(core_sn) >= 2:
                    if core_loc in core_sn or core_sn in core_loc:
                        matched_sheet = sn
                        break
            
            if not matched_sheet:
                # 简单包含匹配
                for sn in sheet_names:
                    if location_str in sn or sn in location_str:
                        matched_sheet = sn
                        break
            
            if not matched_sheet:
                import difflib
                matches = difflib.get_close_matches(location_str, sheet_names, n=1, cutoff=0.3)
                if matches:
                    matched_sheet = matches[0]

        if not matched_sheet:
            # 提示可以用的 sheet 列表，不要直接报错挂掉
            print(f"⚠️ 无法明确匹配输入地点 '{location}'，气象文件中的可用地点有: {sheet_names}")
            # 回退到第一个看起来是有效气象数据的sheet（避免选到“定点监测站点”等说明sheet）
            candidate_sheets = [
                sn for sn in sheet_names
                if ('站点' not in sn and '说明' not in sn and '统计' not in sn)
            ]
            matched_sheet = candidate_sheets[0] if candidate_sheets else sheet_names[0]
            print(f"⚠️ 强制使用首个气象地点 '{matched_sheet}' 进行预测。")
        weather_data = pd.read_excel(weather_file, sheet_name=matched_sheet)
        
        # 确保日期列正确
        date_col = 'date' if 'date' in weather_data.columns else ('日期' if '日期' in weather_data.columns else None)
        if date_col:
            def parse_date(x):
                s = str(x).strip().split('.')[0]
                if len(s) >= 8 and s[:8].isdigit():
                    try:
                        return pd.to_datetime(s[:8], format='%Y%m%d')
                    except ValueError:
                        pass
                return pd.to_datetime(x)
                
            weather_data['date'] = weather_data[date_col].apply(parse_date).dt.normalize()
        
        weather_data = weather_data[
            (weather_data['date'] >= effective_start_date) &
            (weather_data['date'] <= end_date)
        ]
        
        if len(weather_data) == 0:
            raise ValueError(f"指定日期范围内没有气象数据")
        
        # 添加地点列
        weather_data['地点'] = location
        
        # 使用数据预处理器生成派生特征
        preprocessor = CornDiseaseDataPreprocessor(None, None)
        preprocessor.weather_data = weather_data
        processed_data = preprocessor.generate_derived_features()
        processed_data["month"] = pd.to_datetime(processed_data["date"]).dt.month.astype(int)
        processed_data["day_of_year"] = pd.to_datetime(processed_data["date"]).dt.dayofyear.astype(int)
        
        print(f"✓ 气象数据处理完成，共 {len(processed_data)} 条记录")
        
        return processed_data
    
    def predict(self, weather_features, disease_name, target_type='rate'):
        """
        预测单个病害
        
        Args:
            weather_features: 气象特征数据DataFrame
            disease_name: 病害名称（灰斑病/大斑病/白斑病）
            target_type: 目标类型（'rate'=发病株率, 'index'=病情指数）
            
        Returns:
            预测结果
        """
        model_key = f"{disease_name}_{target_type}"
        
        if model_key not in self.models:
            raise ValueError(f"模型 {model_key} 未加载")
        
        model = self.models[model_key]
        scaler = self.scalers[model_key]
        imputer = self.imputers[model_key]
        
        # 获取模型的特征列（优先使用 bundle 中保存的特征列）
        model_features = self.feature_columns_by_model.get(model_key)
        if not model_features:
            model_features = list(getattr(imputer, "feature_names_in_", []))
        if not model_features:
            model_features = list(weather_features.columns)

        # 取出所需的特征。如果实时气象数据中有些特征完全缺失了，我们在 DataFrame 中为其填充为 0 的列
        missing_cols = [col for col in model_features if col not in weather_features.columns]
        if missing_cols:
            print(f"⚠️ 警告: 预测时发现气象数据缺少以下特征: {missing_cols}，已将其填充为缺省值 0。")
            for col in missing_cols:
                weather_features[col] = 0

        # 取出特征并确保顺序与训练时严格一致
        X = weather_features[model_features].copy()

        # 使用插补器填充缺失值（NaN）
        X_imputed = imputer.transform(X)
        
        # 标准化
        X_scaled = scaler.transform(X_imputed)
        
        # 预测
        predictions = model.predict(X_scaled)
        
        return predictions

    def _get_last_actual_state(self, location, actual_file='data/processed_data.csv'):
        if not actual_file or not os.path.exists(actual_file):
            return None
        try:
            df = pd.read_csv(actual_file, encoding='utf-8-sig')
        except Exception:
            return None

        if 'date' not in df.columns or '地点' not in df.columns:
            return None

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df[df['地点'].astype(str).str.strip() == str(location).strip()].copy()
        if len(df) == 0:
            return None

        df = df.sort_values('date')
        last = df.iloc[-1]

        state = {}
        for disease in self.diseases:
            rate_col = f"{disease}发病株率" if f"{disease}发病株率" in df.columns else self.actual_rate_cols.get(disease)
            index_col = f"{disease}病情指数"
            rate_val = float(last.get(rate_col, 0.0)) if rate_col in df.columns else 0.0
            index_val = float(last.get(index_col, 0.0)) if index_col in df.columns else 0.0
            state[disease] = {
                "curr_rate": np.clip(rate_val, 0, 100),
                "curr_index": np.clip(index_val, 0, 100),
            }
        return state

    def predict_all_diseases_recursive(self, weather_features, location=None, actual_file='data/processed_data.csv'):
        """
        使用递归方式进行逐日预测：将上一日预测的累计值作为下一日特征。
        """
        df = weather_features.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['地点', 'date']).reset_index(drop=True)

        all_results = []
        for loc, sub in df.groupby('地点'):
            sub = sub.sort_values('date').reset_index(drop=True)
            init_state = self._get_last_actual_state(loc, actual_file=actual_file) or {}
            curr_rate = {d: init_state.get(d, {}).get('curr_rate', 0.0) for d in self.diseases}
            curr_index = {d: init_state.get(d, {}).get('curr_index', 0.0) for d in self.diseases}
            prev_rate = curr_rate.copy()
            prev_index = curr_index.copy()

            for _, row in sub.iterrows():
                row_base = row.to_frame().T
                row_base['month'] = pd.to_datetime(row_base['date']).dt.month.astype(int)
                row_base['day_of_year'] = pd.to_datetime(row_base['date']).dt.dayofyear.astype(int)

                out_row = {
                    'date': row_base['date'].iloc[0],
                    '地点': loc,
                }

                for disease in self.diseases:
                    row_features = row_base.copy()
                    row_features['curr_rate'] = curr_rate[disease]
                    row_features['curr_index'] = curr_index[disease]
                    row_features['rate_growth_1d'] = curr_rate[disease] - prev_rate[disease]
                    row_features['index_growth_1d'] = curr_index[disease] - prev_index[disease]

                    rate_pred = float(self.predict(row_features, disease, 'rate')[0])
                    index_pred = float(self.predict(row_features, disease, 'index')[0])
                    rate_pred = float(np.clip(rate_pred, 0, 100))
                    index_pred = float(np.clip(index_pred, 0, 100))

                    # 约束：一旦进入非零阶段，后续保持单增（非下降）
                    if curr_rate[disease] > 0 or rate_pred > 0:
                        rate_pred = max(rate_pred, curr_rate[disease])
                    if curr_index[disease] > 0 or index_pred > 0:
                        index_pred = max(index_pred, curr_index[disease])

                    prob_pred = float(self._calculate_disease_probability(rate_pred, index_pred))

                    out_row[f'{disease}_发病概率(%)'] = round(prob_pred, 2)
                    out_row[f'{disease}_风险等级'] = self._probability_to_risk_level(out_row[f'{disease}_发病概率(%)'])

                    prev_rate[disease] = curr_rate[disease]
                    prev_index[disease] = curr_index[disease]
                    curr_rate[disease] = float(rate_pred)
                    curr_index[disease] = float(index_pred)

                disease_prob_cols = [f'{d}_发病概率(%)' for d in self.diseases if f'{d}_发病概率(%)' in out_row]
                if disease_prob_cols:
                    avg_prob = float(np.mean([out_row[c] for c in disease_prob_cols]))
                    out_row['综合发病概率(%)'] = round(avg_prob, 2)
                    out_row['综合风险等级'] = self._probability_to_risk_level(out_row['综合发病概率(%)'])

                all_results.append(out_row)

        return pd.DataFrame(all_results)
    
    def predict_all_diseases(self, weather_features):
        """
        预测所有病害
        
        Args:
            weather_features: 气象特征数据DataFrame
            
        Returns:
            包含所有预测结果的DataFrame
        """
        results = weather_features[['date', '地点']].copy()
        
        for disease in self.diseases:
            # 内部仍预测发病株率和病情指数用于融合概率，但对外仅输出发病概率
            rate_predictions = self.predict(weather_features, disease, 'rate')
            index_predictions = self.predict(weather_features, disease, 'index')
            prob_predictions = self._calculate_disease_probability(rate_predictions, index_predictions)
            results[f'{disease}_发病概率(%)'] = np.round(prob_predictions, 2)
            results[f'{disease}_风险等级'] = self._probability_to_risk_level(results[f'{disease}_发病概率(%)'])

        disease_prob_cols = [f'{d}_发病概率(%)' for d in self.diseases if f'{d}_发病概率(%)' in results.columns]
        if disease_prob_cols:
            results['综合发病概率(%)'] = results[disease_prob_cols].mean(axis=1).round(2)
            results['综合风险等级'] = self._probability_to_risk_level(results['综合发病概率(%)'])
        
        return results
    
    def predict_from_weather_file(self, weather_file, location=None, start_date=None, end_date=None, output_file=None, actual_file='data/processed_data.csv', lookback_days=7, use_recursive_state=True):
        """
        从气象文件预测病害
        
        Args:
            weather_file: 气象数据文件
            location: 地点名称（可选；为空时自动对全部地点预测）
            start_date: 开始日期
            end_date: 结束日期
            output_file: 输出文件路径（可选）
            actual_file: 真实调查数据文件（用于定位首次调查日期）
            lookback_days: 从首次调查日前回看天数（默认7天）
            
        Returns:
            预测结果DataFrame
        """
        # 准备气象特征
        weather_features = self.prepare_weather_features(
            weather_file, location, start_date, end_date,
            actual_file=actual_file,
            lookback_days=lookback_days
        )
        
        # 预测所有病害
        print("\n正在进行病害预测...")
        if use_recursive_state:
            predictions = self.predict_all_diseases_recursive(
                weather_features,
                location=location,
                actual_file=actual_file,
            )
        else:
            predictions = self.predict_all_diseases(weather_features)
        
        # 显示预测结果
        print("\n预测结果预览:")
        print(predictions.head(10))
        
        # 保存到文件
        if output_file:
            predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n预测结果已保存至: {output_file}")
        
        return predictions
    
    def predict_risk_level(self, predictions):
        """
        根据预测结果评估风险等级计算发病可能性

        Args:
            predictions: 预测结果DataFrame

        Returns:
            包含风险等级和可能性的DataFrame
        """
        risk_df = predictions.copy()

        for disease in self.diseases:
            rate_col = f'{disease}_预测发病株率'
            index_col = f'{disease}_预测病情指数'
            risk_col = f'{disease}_风险等级'
            prob_col = f'{disease}_发病可能性(%)'
            prob_src_col = f'{disease}_发病概率(%)'

            # 优先使用预测阶段已产出的发病概率；若不存在则回退到动态计算
            if prob_src_col in risk_df.columns:
                risk_df[prob_col] = risk_df[prob_src_col].clip(0, 100).round(2)
            else:
                risk_df[prob_col] = np.round(
                    self._calculate_disease_probability(risk_df[rate_col], risk_df[index_col]),
                    2
                )

            # 使用发病可能性(%)评估风险等级：低/中/高
            risk_df[risk_col] = self._probability_to_risk_level(risk_df[prob_col])

        # 汇总风险等级（低/中/高）
        prob_cols = [f'{d}_发病可能性(%)' for d in self.diseases if f'{d}_发病可能性(%)' in risk_df.columns]
        if prob_cols:
            risk_df['综合发病可能性(%)'] = risk_df[prob_cols].mean(axis=1).round(2)
            risk_df['综合风险等级'] = self._probability_to_risk_level(risk_df['综合发病可能性(%)'])

        return risk_df

    def generate_forecast_report(self, predictions, output_file='outputs/forecast_report.csv'):
        """
        生成预测报告

        Args:
            predictions: 预测结果DataFrame
            output_file: 输出文件路径
        """
        # 添加风险等级和可能性
        report = self.predict_risk_level(predictions)

        # 统计信息
        print("\n" + "="*60)
        print("💡 病害发病可能性与综合预测报告")
        print("="*60)

        for disease in self.diseases:
            risk_col = f'{disease}_风险等级'
            prob_col = f'{disease}_发病可能性(%)'

            mean_prob = report[prob_col].mean()
            max_prob = report[prob_col].max()
            risk_counts = report[risk_col].value_counts()
            main_risk = risk_counts.idxmax() if not risk_counts.empty else "未知"

            print(f"\n🌽 【{disease}】:")
            print(f"  👉 平均发病可能性: {mean_prob:.1f}%")
            print(f"  👉 最高发病可能性: {max_prob:.1f}% (极值警报)")
            print(f"  📉 综合风险评级: {main_risk}")
            print(f"  日切片风险等级分布:")
            for risk_type, count in risk_counts.items():
                print(f"    - {risk_type}: {count} 天")

        # 保存报告
        report.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n详细预测报告已保存至: {output_file}")
        
        return report


def main():
    """主函数 - 使用示例"""
    print("="*60)
    print("玉米病害预测系统")
    print("="*60)
    
    # 初始化预测器
    forecast = CornDiseaseForecast(models_dir='models', output_root='results')
    
    # 加载模型
    forecast.load_models()
    
    # 配置预测参数
    weather_file = "data/2025年定点监测气象数据.xlsx"
    start_date = "2025-05-01"
    end_date = "2025-09-30"
    
    try:
        # 批量预测11个地点，并输出周尺度图像（一周一个点位）
        all_results, all_comparisons = forecast.batch_predict_all_locations_weekly(
            weather_file=weather_file,
            start_date=start_date,
            end_date=end_date,
            locations=None,
            output_root='results'
        )

        # 输出每种病的所有地点总览大图（预测 vs 真实）
        forecast.plot_disease_overview_all_locations(
            all_comparisons,
            output_dir=forecast.output_dirs['overview_figures']
        )

        # 为每个地点生成报告
        for location, weekly_predictions in all_results.items():
            safe_loc = forecast._sanitize_filename(location)
            forecast.generate_forecast_report(
                weekly_predictions,
                output_file=os.path.join(forecast.output_dirs['reports'], f"forecast_report_{safe_loc}.csv")
            )

        print(f"\n✓ 全部地点预测完成！结果目录: {forecast.output_dirs['root']}")
        
    except Exception as e:
        print(f"\n✗ 预测失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
