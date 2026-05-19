from __future__ import annotations

# [04.1] ===== 基础库导入 =====
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import importlib.util

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

# [04.2] ===== 动态加载配置与特征模块 =====
_cfg_spec = importlib.util.spec_from_file_location("cfg", Path(__file__).with_name("01_config_data_only.py"))
cfg = importlib.util.module_from_spec(_cfg_spec)
assert _cfg_spec and _cfg_spec.loader
_cfg_spec.loader.exec_module(cfg)

_fe_spec = importlib.util.spec_from_file_location("fe", Path(__file__).with_name("03_feature_engineering_data_only.py"))
fe = importlib.util.module_from_spec(_fe_spec)
assert _fe_spec and _fe_spec.loader
_fe_spec.loader.exec_module(fe)


# [04.3] ===== 网络结构 =====
class DiseaseLSTM(nn.Module):
    """双分支网络：LSTM 处理时序，MLP 处理过程特征，最终联合输出双目标增量。"""

    def __init__(self, seq_dim: int, tab_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=seq_dim, hidden_size=20, batch_first=True)
        self.tab_net = nn.Sequential(nn.Linear(tab_dim, 24), nn.ReLU(), nn.Dropout(0.05))
        self.fusion = nn.Sequential(nn.Linear(44, 24), nn.ReLU())
        self.head = nn.Linear(24, 2)

    def extract_penultimate(self, seq_input: torch.Tensor, tab_input: torch.Tensor) -> torch.Tensor:
        seq_output, _ = self.lstm(seq_input)
        seq_hidden = seq_output[:, -1, :]
        tab_hidden = self.tab_net(tab_input)
        return self.fusion(torch.cat([seq_hidden, tab_hidden], dim=1))

    def forward(self, seq_input: torch.Tensor, tab_input: torch.Tensor) -> torch.Tensor:
        penultimate = self.extract_penultimate(seq_input, tab_input)
        return self.head(penultimate)


# [04.4] ===== 训练与损失 =====
def masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    weighted = loss * mask
    denominator = mask.sum().clamp(min=1.0)
    return weighted.sum() / denominator


def split_train_validation_sites(site_ids: list[int], seed: int) -> tuple[list[int], list[int]]:
    shuffled = site_ids[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * 0.2)))
    validation = sorted(shuffled[:val_count])
    training = sorted(shuffled[val_count:])
    if not training:
        training, validation = validation, training
    return training, validation


def train_model(
    site_rows: dict[int, list[dict[str, Any]]],
    train_site_ids: list[int],
    validation_site_ids: list[int],
    targets: list[str],
    seed: int,
) -> dict[str, Any]:
    if xgb is None:
        raise ImportError("当前环境未安装 xgboost，请先安装 xgboost 与 scikit-learn。")

    train_seq, train_tab, train_y, train_mask, train_prev = fe.build_training_arrays(site_rows, train_site_ids, targets)
    val_seq, val_tab, val_y, val_mask, val_prev = fe.build_training_arrays(site_rows, validation_site_ids, targets)

    scalers = fe.fit_scalers(train_seq, train_tab, train_y)
    train_seq_scaled, train_tab_scaled = fe.apply_scalers(train_seq, train_tab, scalers)
    val_seq_scaled, val_tab_scaled = fe.apply_scalers(val_seq, val_tab, scalers)
    train_y_scaled = fe.scale_targets(train_y, scalers)
    val_y_scaled = fe.scale_targets(val_y, scalers)
    train_actual = np.clip(train_prev + train_y, 0.0, 1.0)

    print("train_seq_scaled.shape =", train_seq_scaled.shape)
    print("train_tab_scaled.shape =", train_tab_scaled.shape)
    model = DiseaseLSTM(seq_dim=train_seq_scaled.shape[-1], tab_dim=train_tab_scaled.shape[-1]).to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    train_seq_tensor = torch.tensor(train_seq_scaled, dtype=torch.float32, device=cfg.DEVICE)
    train_tab_tensor = torch.tensor(train_tab_scaled, dtype=torch.float32, device=cfg.DEVICE)
    train_y_tensor = torch.tensor(train_y_scaled, dtype=torch.float32, device=cfg.DEVICE)
    train_mask_tensor = torch.tensor(train_mask, dtype=torch.float32, device=cfg.DEVICE)

    val_seq_tensor = torch.tensor(val_seq_scaled, dtype=torch.float32, device=cfg.DEVICE)
    val_tab_tensor = torch.tensor(val_tab_scaled, dtype=torch.float32, device=cfg.DEVICE)
    val_y_tensor = torch.tensor(val_y_scaled, dtype=torch.float32, device=cfg.DEVICE)
    val_mask_tensor = torch.tensor(val_mask, dtype=torch.float32, device=cfg.DEVICE)

    rng = np.random.default_rng(seed)
    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience = 20
    patience_left = patience
    batch_size = len(train_seq_scaled)

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

    with torch.no_grad():
        train_penultimate = model.extract_penultimate(train_seq_tensor, train_tab_tensor).cpu().numpy()
        val_delta_scaled = model(val_seq_tensor, val_tab_tensor).cpu().numpy()
    val_delta = fe.unscale_targets(val_delta_scaled, scalers)
    val_actual = val_prev + val_y

    train_xgb_features = np.concatenate([train_penultimate, train_tab_scaled], axis=1).astype(np.float32)
    xgb_models: list[Any] = []
    for output_index in range(len(targets)):
        valid_mask = train_mask[:, output_index] > 0.5
        if not np.any(valid_mask):
            xgb_models.append(None)
            continue

        reg = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=getattr(cfg, "XGB_N_ESTIMATORS", 500),
            max_depth=getattr(cfg, "XGB_MAX_DEPTH", 4),
            learning_rate=getattr(cfg, "XGB_LEARNING_RATE", 0.05),
            subsample=getattr(cfg, "XGB_SUBSAMPLE", 0.8),
            colsample_bytree=getattr(cfg, "XGB_COLSAMPLE_BYTREE", 0.8),
            reg_lambda=getattr(cfg, "XGB_REG_LAMBDA", 1.0),
            random_state=seed + output_index,
        )
        reg.fit(train_xgb_features[valid_mask], train_actual[valid_mask, output_index])
        xgb_models.append(reg)

    output_alphas = []
    monotonic_flags = []
    for output_index, target_name in enumerate(targets):
        is_monotonic = target_name in cfg.MONOTONIC_TARGETS
        monotonic_flags.append(is_monotonic)
        valid_mask = val_mask[:, output_index] > 0.5
        if not np.any(valid_mask):
            output_alphas.append(0.0)
            continue
        best_alpha = 0.0
        best_rmse = float("inf")
        for alpha in cfg.ALPHA_GRID:
            pred_values = np.clip(val_prev[valid_mask, output_index] + alpha * val_delta[valid_mask, output_index], 0.0, 1.0)
            if is_monotonic:
                pred_values = np.maximum(pred_values, val_prev[valid_mask, output_index])
            actual_values = np.clip(val_actual[valid_mask, output_index], 0.0, 1.0)
            rmse = float(np.sqrt(np.mean((pred_values - actual_values) ** 2)))
            if rmse < best_rmse - 1e-8:
                best_rmse = rmse
                best_alpha = float(alpha)
        output_alphas.append(best_alpha)

    return {
        "model": model,
        "xgb_models": xgb_models,
        "scalers": scalers,
        "targets": targets,
        "best_val_loss": best_val_loss,
        "output_alphas": np.asarray(output_alphas, dtype=np.float32),
        "monotonic_flags": monotonic_flags,
    }


def train_full_model(site_rows: dict[int, list[dict[str, Any]]], targets: list[str], seed: int) -> dict[str, Any]:
    all_sites = sorted(site_rows.keys())
    train_sites, val_sites = split_train_validation_sites(all_sites, seed)
    return train_model(site_rows, train_sites, val_sites, targets, seed)


# [04.5] ===== 预测与评估 =====
def predict_row(bundle: dict[str, Any], row: dict[str, Any], previous_targets: np.ndarray) -> np.ndarray:
    tab_values = [fe.fill_none(row[name]) for name in cfg.BASE_MODEL_FEATURES] + previous_targets.tolist()
    seq = row["weather_seq_21"].astype(np.float32)
    tab = np.asarray(tab_values, dtype=np.float32)
    seq_scaled, tab_scaled = fe.apply_scalers(seq[None, ...], tab[None, ...], bundle["scalers"])
    seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32, device=cfg.DEVICE)
    tab_tensor = torch.tensor(tab_scaled, dtype=torch.float32, device=cfg.DEVICE)
    with torch.no_grad():
        prediction = bundle["model"](seq_tensor, tab_tensor)
        penultimate = bundle["model"].extract_penultimate(seq_tensor, tab_tensor).cpu().numpy()
    delta_values = fe.unscale_targets(prediction.cpu().numpy(), bundle["scalers"])[0]
    lstm_values = previous_targets + bundle["output_alphas"] * delta_values

    xgb_models = bundle.get("xgb_models", [])
    if xgb_models and len(xgb_models) == len(bundle["targets"]):
        xgb_features = np.concatenate([penultimate[0], tab_scaled[0]], axis=0).astype(np.float32)[None, :]
        xgb_values = []
        for output_index, model in enumerate(xgb_models):
            if model is None:
                xgb_values.append(float(lstm_values[output_index]))
            else:
                xgb_values.append(float(model.predict(xgb_features)[0]))
        values = np.asarray(xgb_values, dtype=np.float32)
    else:
        values = lstm_values.astype(np.float32)

    for output_index, is_monotonic in enumerate(bundle["monotonic_flags"]):
        if is_monotonic:
            values[output_index] = max(values[output_index], previous_targets[output_index])
    return np.clip(values, 0.0, 1.0)

def rolling_predictions(
    site_rows: dict[int, list[dict[str, Any]]],
    bundle: dict[str, Any],
    targets: list[str],
    mode_name: str,
    use_actual_previous: bool = True,
) -> list[dict[str, Any]]:
    prediction_rows = []

    for site_id in sorted(site_rows):
        rows = site_rows[site_id]
        sub_sequences = fe.split_rows_by_replicate(rows)

        for subset in sub_sequences:
            if not subset:
                continue

            replicate_id = subset[0].get("replicate_id_same_day", 1)

            previous_targets = np.array(
                [fe.fill_none(subset[0][targets[0]]) / 100.0, fe.fill_none(subset[0][targets[1]]) / 100.0],
                dtype=np.float32,
            )

            prediction_rows.append(
                {
                    "mode": mode_name,
                    "site_id": site_id,
                    "site_alias": cfg.SITE_ALIAS[site_id],
                    "site_name": subset[0]["site_name"],
                    "date_str": subset[0]["date_str"],
                    "record_id": subset[0]["record_id"],
                    "replicate_id_same_day": replicate_id,
                    f"actual_{targets[0]}": subset[0][targets[0]],
                    f"actual_{targets[1]}": subset[0][targets[1]],
                    f"pred_{targets[0]}": subset[0][targets[0]],
                    f"pred_{targets[1]}": subset[0][targets[1]],
                    "is_baseline": 1,
                }
            )

            for row_index in range(1, len(subset)):
                row = subset[row_index]

                if use_actual_previous:
                    prev_row = subset[row_index - 1]
                    previous_targets = np.array(
                        [fe.fill_none(prev_row[targets[0]]) / 100.0, fe.fill_none(prev_row[targets[1]]) / 100.0],
                        dtype=np.float32,
                    )

                current_pred = predict_row(bundle, row, previous_targets)

                prediction_rows.append(
                    {
                        "mode": mode_name,
                        "site_id": site_id,
                        "site_alias": cfg.SITE_ALIAS[site_id],
                        "site_name": row["site_name"],
                        "date_str": row["date_str"],
                        "record_id": row["record_id"],
                        "replicate_id_same_day": replicate_id,
                        f"actual_{targets[0]}": row[targets[0]],
                        f"actual_{targets[1]}": row[targets[1]],
                        f"pred_{targets[0]}": float(current_pred[0] * 100.0),
                        f"pred_{targets[1]}": float(current_pred[1] * 100.0),
                        "is_baseline": 0,
                    }
                )

                if not use_actual_previous:
                    previous_targets = current_pred.astype(np.float32)

    return prediction_rows
# def rolling_predictions(
#     site_rows: dict[int, list[dict[str, Any]]],
#     bundle: dict[str, Any],
#     targets: list[str],
#     mode_name: str,
#     use_actual_previous: bool = True,
# ) -> list[dict[str, Any]]:
#     prediction_rows = []
#     for site_id in sorted(site_rows):
#         rows = site_rows[site_id]
#         previous_targets = np.array(
#             [fe.fill_none(rows[0][targets[0]]) / 100.0, fe.fill_none(rows[0][targets[1]]) / 100.0], dtype=np.float32
#         )
#         prediction_rows.append(
#             {
#                 "mode": mode_name,
#                 "site_id": site_id,
#                 "site_alias": cfg.SITE_ALIAS[site_id],
#                 "site_name": rows[0]["site_name"],
#                 "date_str": rows[0]["date_str"],

#                 # "survey_order": rows[0]["survey_order"],
#                 "record_id": rows[0]["record_id"],
#                 "replicate_id_same_day": rows[0].get("replicate_id_same_day", 1),

#                 f"actual_{targets[0]}": rows[0][targets[0]],
#                 f"actual_{targets[1]}": rows[0][targets[1]],
#                 f"pred_{targets[0]}": rows[0][targets[0]],
#                 f"pred_{targets[1]}": rows[0][targets[1]],
#                 "is_baseline": 1,
#             }
#         )
#         for row_index, row in enumerate(rows[1:], start=1):
#             if use_actual_previous:
#                 prev_row = rows[row_index - 1]
#                 previous_targets = np.array(
#                     [fe.fill_none(prev_row[targets[0]]) / 100.0, fe.fill_none(prev_row[targets[1]]) / 100.0],
#                     dtype=np.float32,
#                 )
#             current_pred = predict_row(bundle, row, previous_targets)
#             prediction_rows.append(
#                 {
#                     "mode": mode_name,
#                     "site_id": site_id,
#                     "site_alias": cfg.SITE_ALIAS[site_id],
#                     "site_name": row["site_name"],
#                     "date_str": row["date_str"],

#                     # "survey_order": row["survey_order"],
#                     "record_id": row["record_id"],
#                     "replicate_id_same_day": row.get("replicate_id_same_day", 1),

#                     f"actual_{targets[0]}": row[targets[0]],
#                     f"actual_{targets[1]}": row[targets[1]],
#                     f"pred_{targets[0]}": float(current_pred[0] * 100.0),
#                     f"pred_{targets[1]}": float(current_pred[1] * 100.0),
#                     "is_baseline": 0,
#                 }
#             )
#             if not use_actual_previous:
#                 previous_targets = current_pred.astype(np.float32)
#     return prediction_rows


def merge_prediction_tables(prediction_sets: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for rows in prediction_sets:
        for row in rows:
            key = (
                row["mode"], 
                row["site_id"], 
                row["date_str"], 

                # row["survey_order"]
                row.get("replicate_id_same_day", 1),
                row.get("record_id", 0),
            )
            if key not in merged:
                merged[key] = {
                    "mode": row["mode"],
                    "site_id": row["site_id"],
                    "site_alias": row["site_alias"],
                    "site_name": row["site_name"],
                    "date_str": row["date_str"],

                    # "survey_order": row["survey_order"],
                    "record_id": row.get("record_id", 0),
                    "replicate_id_same_day": row.get("replicate_id_same_day", 1),
                    
                    "is_baseline": row["is_baseline"],
                }
            merged[key].update({k: v for k, v in row.items() if k.startswith("actual_") or k.startswith("pred_")})
    return sorted(
        merged.values(), 
        key=lambda item: (
            item["mode"], 
            item["site_id"], 
            item["date_str"],

            # item["survey_order"]
            item.get("replicate_id_same_day", 1),
            item.get("record_id", 0),
        ),
    )


def compute_metrics(prediction_rows: list[dict[str, Any]], mode_name: str) -> list[dict[str, Any]]:
    metrics = []
    target_names = list(cfg.TARGET_LABELS.keys())
    filtered = [row for row in prediction_rows if row["mode"] == mode_name and row["is_baseline"] == 0]
    for target in target_names:
        actual_values = []
        predicted_values = []
        for row in filtered:
            actual = row.get(f"actual_{target}")
            pred = row.get(f"pred_{target}")
            if actual is None or pred is None:
                continue
            actual_values.append(float(actual))
            predicted_values.append(float(pred))
        actual_arr = np.asarray(actual_values, dtype=np.float64)
        pred_arr = np.asarray(predicted_values, dtype=np.float64)
        if len(actual_arr) == 0:
            continue
        mae = float(np.mean(np.abs(pred_arr - actual_arr)))
        rmse = float(np.sqrt(np.mean((pred_arr - actual_arr) ** 2)))
        denominator = float(np.sum((actual_arr - actual_arr.mean()) ** 2))
        r2 = float(1.0 - np.sum((pred_arr - actual_arr) ** 2) / denominator) if denominator > 1e-9 else float("nan")
        metrics.append(
            {
                "mode": mode_name,
                "target": target,
                "target_cn": cfg.TARGET_LABELS[target],
                "n": int(len(actual_arr)),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )
    return metrics
