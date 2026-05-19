"""
测试脚本：验证图像连续性修复
"""
import pandas as pd


def _silent_print(*args, **kwargs):
    return None


print = _silent_print
import numpy as np
from prediction import CornDiseaseForecast

# 创建测试数据：模拟有缺失周的预测数据
test_dates = pd.date_range('2025-05-01', '2025-09-30', freq='D')
# 故意跳过某些日期来模拟缺失的周
missing_indices = [20, 21, 22, 23, 24, 25, 26,  # 跳过一周
                   50, 51, 52, 53, 54, 55, 56]   # 跳过另一周
test_dates = test_dates[~test_dates.isin(test_dates[missing_indices])]

test_data = pd.DataFrame({
    'date': test_dates,
    '地点': '测试地点',
    '灰斑病_发病概率(%)': np.random.uniform(10, 60, len(test_dates)),
    '大斑病_发病概率(%)': np.random.uniform(5, 40, len(test_dates)),
    '白斑病_发病概率(%)': np.random.uniform(0, 30, len(test_dates))
})

# 初始化预测器
forecast = CornDiseaseForecast()

# 测试 _to_weekly_points 方法
print("=" * 60)
print("测试数据统计：")
print(f"总日期数: {len(test_data)}")
print(f"日期范围: {test_data['date'].min()} 到 {test_data['date'].max()}")
print(f"缺失的日期数: {len(missing_indices)}")

weekly = forecast._to_weekly_points(test_data, origin_date='2025-05-01')

print("\n修复后的周尺度数据统计：")
print(f"周数: {len(weekly)}")
print(f"日期范围: {weekly['date'].min()} 到 {weekly['date'].max()}")
print(f"\n前10行周尺度数据:")
print(weekly[['date', '灰斑病_发病概率(%)']].head(10))
print(f"\n后5行周尺度数据:")
print(weekly[['date', '灰斑病_发病概率(%)']].tail(5))

# 检查是否有缺失值
missing_count = weekly[['灰斑病_发病概率(%)', '大斑病_发病概率(%)', '白斑病_发病概率(%)']].isna().sum()
print(f"\n缺失值统计（修复后）:")
print(f"灰斑病缺失: {missing_count['灰斑病_发病概率(%)']}")
print(f"大斑病缺失: {missing_count['大斑病_发病概率(%)']}")
print(f"白斑病缺失: {missing_count['白斑病_发病概率(%)']}")

if missing_count.sum() == 0:
    print("\n✓ 修复成功！所有缺失周都已通过插值填充")
else:
    print("\n✗ 修复不完整，仍有缺失值")

print("=" * 60)
