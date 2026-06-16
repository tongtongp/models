from __future__ import annotations

from pathlib import Path
import torch
import importlib.util
import pandas as pd

from data_pipeline import build_training_records, load_site_rows
from manba_model import ensure_mamba_available
from training import (
    build_prediction_table,
    compute_metrics,
    predict_records,
    save_bundle,
    train_full_model,
)

_BASE_DIR = Path(__file__).resolve().parent
_XLSTM_DIR = _BASE_DIR.parent / "xgboost+lstm"

_cfg_spec = importlib.util.spec_from_file_location("cfg", _XLSTM_DIR / "01_config.py")
cfg = importlib.util.module_from_spec(_cfg_spec)
assert _cfg_spec and _cfg_spec.loader
_cfg_spec.loader.exec_module(cfg)


def main() -> None:
    if not ensure_mamba_available():
        print("提示：未检测到 mamba-ssm，将使用纯 PyTorch SSM 退化实现（兼容 Python 3.13）。")

    outputs_dir = _BASE_DIR / "outputs"
    models_dir = _BASE_DIR / "models"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    cfg.set_global_seed(cfg.RANDOM_SEED)

    site_rows = load_site_rows()

    summary_rows = []

    for disease_key, disease_conf in cfg.DISEASE_CONFIGS.items():
        targets = disease_conf["targets"]

        bundle = train_full_model(site_rows, targets, cfg.RANDOM_SEED + 100)
        bundle_path = models_dir / f"manba_bundle_{disease_key}.pt"
        save_bundle(bundle, bundle_path)

        all_site_ids = sorted(site_rows.keys())
        records = build_training_records(site_rows, all_site_ids, targets)
        preds, actual, mask = predict_records(bundle, records)

        pred_table = build_prediction_table(records, preds, actual, mask, targets)
        pred_path = outputs_dir / f"predictions_{disease_key}.csv"
        pred_table.to_csv(pred_path, index=False, encoding="utf-8-sig")

        metrics = compute_metrics(preds, actual, mask, targets)
        metrics_path = outputs_dir / f"metrics_{disease_key}.csv"
        metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

        summary_rows.append(
            {
                "disease": disease_key,
                "bundle_path": str(bundle_path),
                "predictions": str(pred_path),
                "metrics": str(metrics_path),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outputs_dir / "run_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
