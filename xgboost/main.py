"""
精简后的主程序：
1. 数据预处理
2. 训练 XGBoost 模型
3. 生成在线滚动预测可直接使用的 xg_full_bundle_*.pt

说明：
- 本脚本不再调用旧的离线预测类 CornDiseaseForecast。
- 未来 7 天预测请走现有在线链路，并传 model_type="XGBoost"。
"""

from __future__ import annotations

import os

from data_preprocessing import CornDiseaseDataPreprocessor
from model_training import CornDiseasePredictor


def _silent_print(*args, **kwargs):
    return None


print = _silent_print


def main() -> None:
    print("\n" + "=" * 70)
    print(" " * 18 + "玉米病害 XGBoost 训练与打包系统")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    disease_file = os.path.join(script_dir, "data", "2025定点监测叶斑病调查数据.xlsx")
    weather_file = os.path.join(script_dir, "data", "2025年定点监测气象数据.xlsx")

    # 将所有运行输出放到对应模型目录下，便于管理多模型产物
    base_model_dir = os.path.join(script_dir, "models", "Xgboost")
    processed_file = os.path.join(base_model_dir, "processed_data.csv")
    outputs_root = os.path.join(base_model_dir, "outputs")

    os.makedirs(os.path.join(script_dir, "models"), exist_ok=True)
    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(outputs_root, exist_ok=True)

    if not os.path.exists(disease_file):
        print(f"错误: 找不到病害数据文件 {disease_file}")
        return
    if not os.path.exists(weather_file):
        print(f"错误: 找不到气象数据文件 {weather_file}")
        return

    print("\n【步骤 1/2】数据预处理")
    print("-" * 70)
    try:
        preprocessor = CornDiseaseDataPreprocessor(disease_file, weather_file)
        preprocessor.process_all(
            output_file=processed_file,
            feature_file=os.path.join(base_model_dir, "xgb_feature_columns.txt"),
        )
        preprocessor.save_feature_target_correlations(
            output_dir=os.path.join(outputs_root, "figures", "feature_importance")
        )
        print("\n✓ 数据预处理完成")
    except Exception as exc:
        print(f"\n✗ 数据预处理失败: {exc}")
        import traceback
        traceback.print_exc()
        return

    print("\n【步骤 2/2】训练 XGBoost 并打包 bundle")
    print("-" * 70)
    try:
        predictor = CornDiseasePredictor(processed_file, outputs_root=outputs_root)
        predictor.load_data()
        predictor.train_all_models()
        print("\n✓ 模型训练与 bundle 打包完成")
    except Exception as exc:
        print(f"\n✗ 模型训练失败: {exc}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 70)
    print("全部完成")
    print("=" * 70)
    print("\n生成文件：")
    print(f"  1. {processed_file} - 预处理后的训练数据")
    print(f"  2. {os.path.join(base_model_dir, 'xgb_feature_columns.txt')} - 训练特征列")
    print(f"  3. {os.path.join(base_model_dir, 'model_*_*.pkl')} - 单目标 XGBoost 模型")
    print(f"  4. {os.path.join(base_model_dir, 'scaler_*_*.pkl')} - 标准化器")
    print(f"  5. {os.path.join(base_model_dir, 'imputer_*_*.pkl')} - 缺失值填补器")
    print(f"  6. {os.path.join(base_model_dir, 'xg_full_bundle_gray.pt')}")
    print(f"  7. {os.path.join(base_model_dir, 'xg_full_bundle_blight.pt')}")
    print(f"  8. {os.path.join(base_model_dir, 'xg_full_bundle_white.pt')}")
    print("\n在线未来 7 天预测时，请在现有在线链路中传入 model_type='XGBoost'。")


if __name__ == "__main__":
    main()
