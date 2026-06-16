from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import importlib.util
import numpy as np
import pandas as pd
import torch
from torch import nn

from data_pipeline import TrainingRecord, build_training_records
from manba_model import DiseaseMamba

_BASE_DIR = Path(__file__).resolve().parent
_XLSTM_DIR = _BASE_DIR.parent / "xgboost+lstm"

_cfg_spec = importlib.util.spec_from_file_location("cfg", _XLSTM_DIR / "01_config.py")
cfg = importlib.util.module_from_spec(_cfg_spec)
assert _cfg_spec and _cfg_spec.loader
_cfg_spec.loader.exec_module(cfg)

_fe_spec = importlib.util.spec_from_file_location("fe", _XLSTM_DIR / "03_feature_engineering.py")
fe = importlib.util.module_from_spec(_fe_spec)
assert _fe_spec and _fe_spec.loader
_fe_spec.loader.exec_module(fe)


def masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    weighted = loss * mask
    denominator = mask.sum().clamp(min=1.0)
    return weighted.sum() / denominator


def split_train_validation_sites(site_ids: list[int], seed: int) -> tuple[list[int], list[int]]:
    shuffled = site_ids[:]
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * 0.2)))
    validation = sorted(shuffled[:val_count])
    training = sorted(shuffled[val_count:])
    if not training:
        training, validation = validation, training
    return training, validation


def _records_to_arrays(records: list[TrainingRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seqs = np.stack([r.seq for r in records]).astype(np.float32)
    tabs = np.stack([r.tab for r in records]).astype(np.float32)
    ys = np.stack([r.y for r in records]).astype(np.float32)
    masks = np.stack([r.mask for r in records]).astype(np.float32)
    prev_targets = np.stack([r.prev_targets for r in records]).astype(np.float32)
    return seqs, tabs, ys, masks, prev_targets


def train_model(
    site_rows: dict[int, list[dict[str, Any]]],
    train_site_ids: list[int],
    validation_site_ids: list[int],
    targets: list[str],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    train_records = build_training_records(site_rows, train_site_ids, targets)
    val_records = build_training_records(site_rows, validation_site_ids, targets)
    if not train_records:
        raise ValueError("训练样本为空，请检查输入数据是否齐全。")

    train_seq, train_tab, train_y, train_mask, train_prev = _records_to_arrays(train_records)
    val_seq, val_tab, val_y, val_mask, val_prev = _records_to_arrays(val_records)

    scalers = fe.fit_scalers(train_seq, train_tab, train_y)
    train_seq_scaled, train_tab_scaled = fe.apply_scalers(train_seq, train_tab, scalers)
    val_seq_scaled, val_tab_scaled = fe.apply_scalers(val_seq, val_tab, scalers)
    train_y_scaled = fe.scale_targets(train_y, scalers)
    val_y_scaled = fe.scale_targets(val_y, scalers)

    model = DiseaseMamba(
        seq_dim=train_seq_scaled.shape[-1],
        tab_dim=train_tab_scaled.shape[-1],
        d_model=64,
        n_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_seq_tensor = torch.tensor(train_seq_scaled, dtype=torch.float32, device=device)
    train_tab_tensor = torch.tensor(train_tab_scaled, dtype=torch.float32, device=device)
    train_y_tensor = torch.tensor(train_y_scaled, dtype=torch.float32, device=device)
    train_mask_tensor = torch.tensor(train_mask, dtype=torch.float32, device=device)

    val_seq_tensor = torch.tensor(val_seq_scaled, dtype=torch.float32, device=device)
    val_tab_tensor = torch.tensor(val_tab_scaled, dtype=torch.float32, device=device)
    val_y_tensor = torch.tensor(val_y_scaled, dtype=torch.float32, device=device)
    val_mask_tensor = torch.tensor(val_mask, dtype=torch.float32, device=device)

    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience = 20
    patience_left = patience
    batch_size = len(train_seq_scaled)

    rng = np.random.default_rng(seed)

    for _ in range(120):
        permutation = rng.permutation(len(train_seq_scaled))
        model.train()
        for start in range(0, len(permutation), batch_size):
            batch_ids = permutation[start : start + batch_size]
            pred = model(train_seq_tensor[batch_ids], train_tab_tensor[batch_ids])
            loss = masked_smooth_l1(pred, train_y_tensor[batch_ids], train_mask_tensor[batch_ids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_seq_tensor, val_tab_tensor)
            val_loss = float(masked_smooth_l1(val_pred, val_y_tensor, val_mask_tensor).cpu().item())

        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()

    monotonic_flags = [target in cfg.MONOTONIC_TARGETS for target in targets]

    return {
        "model": model,
        "scalers": scalers,
        "targets": targets,
        "best_val_loss": best_val_loss,
        "monotonic_flags": monotonic_flags,
    }


def train_full_model(site_rows: dict[int, list[dict[str, Any]]], targets: list[str], seed: int) -> dict[str, Any]:
    site_ids = sorted(site_rows.keys())
    train_sites, val_sites = split_train_validation_sites(site_ids, seed)
    return train_model(site_rows, train_sites, val_sites, targets, seed, cfg.DEVICE)


def predict_records(
    bundle: dict[str, Any],
    records: list[TrainingRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("预测样本为空，请检查输入数据。")

    seq, tab, y, mask, prev = _records_to_arrays(records)
    scalers = bundle["scalers"]

    seq_scaled, tab_scaled = fe.apply_scalers(seq, tab, scalers)

    model: DiseaseMamba = bundle["model"]
    model.eval()

    with torch.no_grad():
        seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32, device=cfg.DEVICE)
        tab_tensor = torch.tensor(tab_scaled, dtype=torch.float32, device=cfg.DEVICE)
        delta_scaled = model(seq_tensor, tab_tensor).cpu().numpy()

    delta = fe.unscale_targets(delta_scaled, scalers)
    preds = prev + delta

    for idx, is_mono in enumerate(bundle.get("monotonic_flags", [False, False])):
        if is_mono:
            preds[:, idx] = np.maximum(preds[:, idx], prev[:, idx])

    preds = np.clip(preds, 0.0, 1.0)
    actual = np.clip(prev + y, 0.0, 1.0)
    return preds, actual, mask


def build_prediction_table(
    records: list[TrainingRecord],
    preds: np.ndarray,
    actual: np.ndarray,
    mask: np.ndarray,
    targets: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec, pred_vals, actual_vals, mask_vals in zip(records, preds, actual, mask):
        row = {
            "site_id": rec.site_id,
            "date": rec.date_value,
        }
        for idx, target in enumerate(targets):
            row[f"actual_{target}"] = actual_vals[idx] * 100.0 if mask_vals[idx] > 0.5 else np.nan
            row[f"pred_{target}"] = pred_vals[idx] * 100.0 if mask_vals[idx] > 0.5 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_metrics(preds: np.ndarray, actual: np.ndarray, mask: np.ndarray, targets: list[str]) -> pd.DataFrame:
    rows = []
    for idx, target in enumerate(targets):
        valid = mask[:, idx] > 0.5
        if not np.any(valid):
            rows.append({"target": target, "rmse": np.nan, "mae": np.nan})
            continue
        diff = preds[valid, idx] - actual[valid, idx]
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        rows.append({"target": target, "rmse": rmse, "mae": mae})
    return pd.DataFrame(rows)


def save_bundle(bundle: dict[str, Any], save_path: Path) -> None:
    model: DiseaseMamba = bundle["model"]
    payload = {
        "seq_dim": int(model.seq_proj.in_features),
        "tab_dim": int(model.tab_net[0].in_features),
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scalers": bundle["scalers"],
        "targets": list(bundle["targets"]),
        "best_val_loss": float(bundle["best_val_loss"]),
        "monotonic_flags": list(bundle["monotonic_flags"]),
        "seq_features": list(cfg.SEQ_FEATURES),
        "base_model_features": list(cfg.BASE_MODEL_FEATURES),
        "lookback_days": int(cfg.LOOKBACK_DAYS),
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_path)


def load_bundle(save_path: Path) -> dict[str, Any]:
    payload = torch.load(save_path, map_location=cfg.DEVICE, weights_only=False)
    model = DiseaseMamba(
        seq_dim=int(payload["seq_dim"]),
        tab_dim=int(payload["tab_dim"]),
    ).to(cfg.DEVICE)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "scalers": payload["scalers"],
        "targets": payload["targets"],
        "best_val_loss": payload["best_val_loss"],
        "monotonic_flags": payload["monotonic_flags"],
    }
