# Manba (Mamba-SSM) 玉米病害预测

本目录新增基于 Mamba（SSM） 的双目标回归模型，用于预测病情指数与发病株率。输入特征与字段来源于 `xgboost+lstm/03_feature_engineering.py` 产生的序列与过程特征。

## 目录结构

- `main.py`：训练与预测入口
- `manba_model.py`：Mamba 模型结构
- `data_pipeline.py`：数据加载与训练样本构建（复用 xgboost+lstm 特征工程）
- `training.py`：训练、预测、指标计算与 bundle 保存
- `models/`：模型产物
- `outputs/`：预测结果与指标

## 输出

- `models/manba_bundle_{disease}.pt`：每个病害一个模型包
- `outputs/predictions_{disease}.csv`：全量预测结果
- `outputs/metrics_{disease}.csv`：RMSE/MAE 指标
- `outputs/run_summary.csv`：运行汇总

## 运行

1. 安装依赖（建议在虚拟环境内）：

```bash
pip install -r requirements.txt
```

2. 训练 + 预测：

```bash
python main.py
```

## 说明

- 训练标签为“本次调查相对上次调查的增量”，预测时会叠加回上一期值并限制在 0~100。
- 若缺失标签，将自动跳过该样本的损失计算与指标统计。
