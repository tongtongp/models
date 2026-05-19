# 玉米病害预测系统

基于XGBoost的玉米病害（灰斑病、大斑病、白斑病）发病率预测系统。通过气象数据预测玉米病害的发病株率和病情指数。

## 项目结构

```
Corn1/
├── data/                    # 数据目录，存放原始数据和处理后的CSV
├── models/                  # 模型目录，存放XGBoost模型及标准化器
├── outputs/                 # 输出目录，存放结果图表及预测报告
├── data_preprocessing.py    # 数据预处理模块
├── model_training.py        # 模型训练模块
├── prediction.py            # 病害预测模块
├── main.py                  # 主程序（完整流程）
├── requirements.txt         # Python依赖包
└── README.md                # 项目说明文档
```

## 功能特点

### 1. 数据预处理
- ✅ 按地点匹配气象数据和病害调查数据
- ✅ 温度单位转换（K → ℃）
- ✅ 相对湿度计算
- ✅ 弱风日连续天数统计
- ✅ 连续降雨天数统计
- ✅ 时间窗口特征（7天、14天滑动平均）
- ✅ 病害易发条件指标（高温高湿、适温高湿）

### 2. 派生特征

**温度特征：**
- 气温平均、最高、最低（℃）
- 气温日较差
- 地表温度
- 土壤温度（5cm）

**湿度特征：**
- 相对湿度平均、最高、最低（%）
- 土壤湿度
- 土壤相对湿度

**风速特征：**
- 风速平均、最高、最低
- 弱风日判定（<3m/s）
- 弱风日连续天数

**降水特征：**
- 24小时降水量
- 连续降雨天数
- 过去7天、14天累计降雨

**时间窗口特征：**
- 过去7天/14天平均气温、湿度、风速
- 过去7天/14天累计降雨量

**病害易发条件：**
- 高温高湿天（>25℃ & >80%湿度）
- 适温高湿天（20-30℃ & >70%湿度）

### 3. XGBoost模型训练
- 为三种病害分别训练模型
- 每种病害训练两个模型：发病株率 + 病情指数
- 共6个模型
- 包含交叉验证和特征重要性分析

### 4. 病害预测
- 基于气象数据预测未来病害发病概率
- 风险等级评估（低、中低、中等、中高、高）
- 生成详细预测报告

## 数据要求

### 病害调查数据（2025定点监测叶斑病调查数据.xlsx）
| 字段名 | 说明 |
|--------|------|
| 序号 | 记录编号 |
| 时间 | 调查日期 |
| 地点 | 调查地点 |
| 品种 | 玉米品种 |
| 灰斑病抗性 | 品种抗性 |
| 大斑病抗性 | 品种抗性 |
| 白斑病抗性 | 品种抗性 |
| 生育期 | 玉米生育期 |
| 灰斑病发病株率 | 目标变量 |
| 灰斑病病情指数 | 目标变量 |
| 大斑病发病株率 | 目标变量 |
| 大斑病病情指数 | 目标变量 |
| 白斑病发病株率 | 目标变量 |
| 白斑病病情指数 | 目标变量 |

### 气象数据（2025年定点监测气象数据.xlsx）
- **文件格式**：每个地点一个sheet
- **数据频率**：每天一条记录

| 类别 | 字段名 | 单位 |
|------|--------|------|
| 基础 | date | - |
| 基础 | LAT(degrees_north) | 度 |
| 基础 | LON(degrees_east) | 度 |
| 气温 | 2m气温平均值/最大值/最小值 | K |
| 地表温度 | 地表温度平均值/最大值/最小值 | K |
| 土壤温度 | 5cm土壤温度 | K |
| 土壤湿度 | 土壤湿度 | m³/m³ |
| 土壤湿度 | 0–10cm土壤相对湿度 | % |
| 空气湿度 | 2m比湿平均值/最大值/最小值 | kg/kg |
| 风速 | 10m风速平均值/最大值/最小值 | m/s |
| 气压 | 地面气压平均值/最大值/最小值 | Pa |
| 降水 | 24小时内降水量之和/最大/最小 | mm |
| 辐射 | 短波辐射平均值/最大值/最小值 | W/m² |

## 安装依赖

```bash
pip install -r requirements.txt
```

### 依赖包列表
- pandas >= 2.0.0
- numpy >= 1.24.0
- openpyxl >= 3.1.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- joblib >= 1.3.0

## 使用方法

### 方法1：运行完整流程（推荐）

```bash
python main.py
```

这将按顺序执行：
1. 数据预处理
2. 模型训练
3. 病害预测

### 方法2：分步执行

#### 步骤1：数据预处理
```python
from data_preprocessing import CornDiseaseDataPreprocessor

preprocessor = CornDiseaseDataPreprocessor(
    disease_file="2025定点监测叶斑病调查数据.xlsx",
    weather_file="2025年定点监测气象数据.xlsx"
)

processed_data = preprocessor.process_all(output_file='processed_data.csv')
```

#### 步骤2：模型训练
```python
from model_training import CornDiseasePredictor

predictor = CornDiseasePredictor('processed_data.csv')
predictor.load_data()
results = predictor.train_all_models()
```

#### 步骤3：病害预测
```python
from prediction import CornDiseaseForecast

forecast = CornDiseaseForecast(models_dir='.')
forecast.load_models()

predictions = forecast.predict_from_weather_file(
    weather_file="2025年定点监测气象数据.xlsx",
    location="北京",  # 地点名称（对应Excel中的sheet名）
    start_date="2025-05-01",
    end_date="2025-09-30",
    output_file="predictions_beijing.csv"
)

report = forecast.generate_forecast_report(predictions)
```

## 输出文件

### 数据预处理阶段
- `processed_data.csv` - 预处理后的合并数据

### 模型训练阶段
- `model_灰斑病_rate.pkl` - 灰斑病发病株率预测模型
- `model_灰斑病_index.pkl` - 灰斑病病情指数预测模型
- `model_大斑病_rate.pkl` - 大斑病发病株率预测模型
- `model_大斑病_index.pkl` - 大斑病病情指数预测模型
- `model_白斑病_rate.pkl` - 白斑病发病株率预测模型
- `model_白斑病_index.pkl` - 白斑病病情指数预测模型
- `scaler_*.pkl` - 对应的数据标准化器
- `model_evaluation_summary.csv` - 模型评估指标汇总
- `feature_importance_*.png` - 特征重要性图表
- `prediction_results_*.png` - 预测结果对比图

### 预测阶段
- `predictions_{地点}.csv` - 预测结果（包含日期、发病率等）
- `forecast_report_{地点}.csv` - 详细预测报告（包含风险等级）

## 模型评估指标

- **RMSE** (Root Mean Square Error) - 均方根误差
- **MAE** (Mean Absolute Error) - 平均绝对误差
- **R²** (R-squared) - 决定系数
- **交叉验证得分** - 5折交叉验证RMSE

## 风险等级标准

| 风险等级 | 发病株率 | 病情指数 |
|---------|---------|---------|
| 低风险 | <10% | <5 |
| 中低风险 | <30% | <15 |
| 中等风险 | <50% | <30 |
| 中高风险 | <70% | <50 |
| 高风险 | ≥70% | ≥50 |

## 注意事项

1. **数据文件名**：确保Excel文件名与代码中一致
2. **地点匹配**：气象数据的sheet名称必须与病害数据中的"地点"字段一致
3. **日期格式**：确保日期列格式正确（YYYY-MM-DD或Excel日期格式）
4. **缺失值处理**：程序会自动处理缺失值，但建议提前检查数据质量
5. **中文显示**：如遇图表中文显示问题，请确保系统安装了中文字体

## 常见问题

### Q1: 提示"找不到气象数据"
**A**: 检查Excel文件中的sheet名称是否与指定的地点名称完全一致（包括空格）

### Q2: 模型训练时间过长
**A**: 可以调整XGBoost参数中的`n_estimators`（默认200），减小可加快训练

### Q3: 预测结果异常
**A**: 检查气象数据的时间范围是否覆盖预测时段，且数据质量良好

### Q4: 内存不足
**A**: 可以分批处理数据，或减少时间窗口特征的计算周期

## 技术支持

如有问题或建议，请联系技术支持团队。

## 版本信息

- **版本**: 1.0.0
- **更新日期**: 2025年
- **Python版本**: 3.8+

## 许可证

本项目仅供学习和研究使用。
