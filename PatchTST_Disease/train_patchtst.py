import ast
import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from patchtst_model import DiseasePatchTSTFusion


PROJECT_DIR = r"C:\Users\tong2\Desktop\models\PatchTST_Disease"
DATA_PATH = os.path.join(PROJECT_DIR, "data", "corn_disease.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_COL = "weather_seq_28"

TARGET_COLS = [
    "gray_incidence",
    "gray_index",
    "blight_incidence",
    "blight_index",
    "white_incidence",
    "white_index",
]

DISEASE_GROUPS = {
    "gray": ["gray_incidence", "gray_index"],
    "blight": ["blight_incidence", "blight_index"],
    "white": ["white_incidence", "white_index"],
}

PREV_TARGET_COLS = [f"prev_{col}" for col in TARGET_COLS]
DELTA_TARGET_COLS = [f"delta_{col}" for col in TARGET_COLS]

BASE_TAB_FEATURE_COLS = [
    "gdd_cum",
    "stage_code",
    "rain_21d_sum",
    "rain_7d_sum",
    "rain_14d_sum",
    "rain_28d_sum",
    "rainy_streak_days",
    "rain_gap_days",
    "temp_21d_mean",
    "temp_7d_mean",
    "temp_14d_mean",
    "temp_28d_mean",
    "temp_range_24h_c",
    "rh_21d_mean",
    "rh_7d_mean",
    "rh_14d_mean",
    "rh_28d_mean",
    "humidity_range_daily",
    "soil_rel_humidity_14d_mean",
    "soil_rel_humidity_7d_mean",
    "soil_rel_humidity_21d_mean",
    "soil_rel_humidity_28d_mean",
    "wind_7d_mean",
    "wind_28d_mean",
    "is_weak_wind_day",
    "weak_wind_streak_days",
    "radiation_7d_mean",
    "radiation_28d_mean",
    "low_radiation_streak_days",
    "hot_streak_days",
    "cold_streak_days",
    "optimal_temp_streak_days",
    "high_humidity_streak_days",
    "high_humidity_7d_count",
    "high_humidity_28d_count",
    "heavy_rain_7d_count",
    "heavy_rain_28d_count",
    "heavy_rain_streak_days",
    "max_single_day_rain_7d",
    "max_single_day_rain_28d",
    "hot_humid_streak_days",
    "optimal_temp_humid_streak_days",
    "weak_wind_humid_streak_days",
]

CONFIG = {
    "seq_len": 28,
    "patch_len": 7,
    "stride": 3,
    "d_model": 32,
    "n_heads": 4,
    "e_layers": 1,
    "d_ff": 64,
    "dropout": 0.25,
    "batch_size": 16,
    "epochs": 200,
    "learning_rate": 5e-4,
    "weight_decay": 1e-3,
    "patience": 35,
    "split_mode": "within_site_time",
    "seed": 42,
}


class DiseaseDataset(torch.utils.data.Dataset):
    def __init__(self, x_seq, x_tab, y):
        self.x_seq = torch.tensor(x_seq, dtype=torch.float32)
        self.x_tab = torch.tensor(x_tab, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x_seq[index], self.x_tab[index], self.y[index]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_weather_seq(value):
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, list):
        arr = np.asarray(value, dtype=np.float32)
    elif isinstance(value, str):
        text = value.strip()
        try:
            arr = np.asarray(json.loads(text), dtype=np.float32)
        except Exception:
            arr = np.asarray(ast.literal_eval(text), dtype=np.float32)
    else:
        raise ValueError(f"Cannot parse {SEQ_COL}: {type(value)}")

    if arr.ndim != 2:
        raise ValueError(f"{SEQ_COL} must be a 2D array, got shape={arr.shape}")

    return arr.astype(np.float32)


def build_split_masks(df, split_mode, seed):
    if split_mode == "site":
        site_ids = sorted(df["site_id"].dropna().unique().tolist())
        rng = np.random.default_rng(seed)
        rng.shuffle(site_ids)

        n_site = len(site_ids)
        train_sites = set(site_ids[: int(n_site * 0.7)])
        val_sites = set(site_ids[int(n_site * 0.7) : int(n_site * 0.85)])
        test_sites = set(site_ids[int(n_site * 0.85) :])

        return (
            df["site_id"].isin(train_sites).values,
            df["site_id"].isin(val_sites).values,
            df["site_id"].isin(test_sites).values,
        )

    if split_mode != "within_site_time":
        raise ValueError(f"Unknown split_mode: {split_mode}")

    group_col = "disease_series_id" if "disease_series_id" in df.columns else "site_id"
    sort_cols = [group_col, "date" if "date" in df.columns else "date_str"]

    train_mask = np.zeros(len(df), dtype=bool)
    val_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)

    for _, group_df in df.sort_values(sort_cols).groupby(group_col, sort=False):
        indices = group_df.index.to_numpy()
        n = len(indices)
        if n < 3:
            train_mask[indices] = True
            continue

        train_end = max(1, int(n * 0.7))
        val_end = max(train_end + 1, int(n * 0.85))
        if val_end >= n:
            val_end = n - 1

        train_mask[indices[:train_end]] = True
        val_mask[indices[train_end:val_end]] = True
        test_mask[indices[val_end:]] = True

    return train_mask, val_mask, test_mask


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for x_seq, x_tab, y in loader:
        x_seq = x_seq.to(device)
        x_tab = x_tab.to(device)
        y = y.to(device)

        pred = model(x_seq, x_tab)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x_seq.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds = []
    trues = []

    with torch.no_grad():
        for x_seq, x_tab, y in loader:
            x_seq = x_seq.to(device)
            x_tab = x_tab.to(device)
            y = y.to(device)

            pred = model(x_seq, x_tab)
            loss = criterion(pred, y)

            total_loss += loss.item() * x_seq.size(0)
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())

    return total_loss / len(loader.dataset), np.concatenate(preds), np.concatenate(trues)


def metric_report(y_true, y_pred, target_cols):
    report = {}
    for i, col in enumerate(target_cols):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        report[col] = {
            "MAE": float(mean_absolute_error(true_i, pred_i)),
            "RMSE": float(mean_squared_error(true_i, pred_i) ** 0.5),
            "R2": float(r2_score(true_i, pred_i)),
        }
    return report


def split_summary(df, masks, target_cols):
    summary = {}
    for name, mask in masks.items():
        part = df.loc[mask, target_cols]
        summary[name] = {
            "rows": int(mask.sum()),
            "sites": int(df.loc[mask, "site_id"].nunique()),
            "series": int(df.loc[mask, "disease_series_id"].nunique())
            if "disease_series_id" in df.columns
            else None,
            "target_mean": {col: float(part[col].mean()) for col in target_cols},
            "target_std": {col: float(part[col].std(ddof=0)) for col in target_cols},
        }
    return summary


def validate_required_columns(df):
    required = [SEQ_COL, "site_id", *BASE_TAB_FEATURE_COLS]
    required.extend(TARGET_COLS)
    required.extend(PREV_TARGET_COLS)
    required.extend(DELTA_TARGET_COLS)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "corn_disease.csv is missing columns: "
            + ", ".join(missing)
            + ". Run prepare_data.py first."
        )


def train_disease_model(df, disease_name, target_cols, device):
    output_dir = os.path.join(OUTPUT_DIR, disease_name)
    os.makedirs(output_dir, exist_ok=True)

    prev_cols = [f"prev_{col}" for col in target_cols]
    delta_cols = [f"delta_{col}" for col in target_cols]
    tab_feature_cols = [*BASE_TAB_FEATURE_COLS, *prev_cols]

    work_df = df.dropna(subset=[*target_cols, *prev_cols, *delta_cols]).copy()
    for col in [*tab_feature_cols, *target_cols, *prev_cols, *delta_cols]:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
    work_df[tab_feature_cols] = work_df[tab_feature_cols].ffill().bfill().fillna(0)
    work_df = work_df.dropna(subset=[*target_cols, *prev_cols, *delta_cols]).reset_index(drop=True)

    x_seq = np.stack([parse_weather_seq(v) for v in work_df[SEQ_COL].values], axis=0)
    x_tab = work_df[tab_feature_cols].values.astype(np.float32)
    prev_y = work_df[prev_cols].values.astype(np.float32)
    y_current = work_df[target_cols].values.astype(np.float32)
    y_delta = work_df[delta_cols].values.astype(np.float32)

    seq_len = x_seq.shape[1]
    seq_feature_dim = x_seq.shape[2]
    tab_feature_dim = x_tab.shape[1]
    CONFIG["seq_len"] = int(seq_len)

    train_mask, val_mask, test_mask = build_split_masks(
        work_df,
        CONFIG["split_mode"],
        CONFIG["seed"],
    )
    masks = {"train": train_mask, "val": val_mask, "test": test_mask}
    split_info = split_summary(work_df, masks, target_cols)

    x_seq_train, x_tab_train, y_train = x_seq[train_mask], x_tab[train_mask], y_delta[train_mask]
    x_seq_val, x_tab_val, y_val = x_seq[val_mask], x_tab[val_mask], y_delta[val_mask]
    x_seq_test, x_tab_test, y_test = x_seq[test_mask], x_tab[test_mask], y_delta[test_mask]
    prev_y_test = prev_y[test_mask]
    y_current_test = y_current[test_mask]

    if len(x_seq_train) == 0 or len(x_seq_val) == 0 or len(x_seq_test) == 0:
        raise ValueError(f"{disease_name}: train/val/test split contains an empty set.")

    seq_scaler = StandardScaler()
    tab_scaler = StandardScaler()
    target_scaler = StandardScaler()

    seq_scaler.fit(x_seq_train.reshape(-1, seq_feature_dim))
    tab_scaler.fit(x_tab_train)
    target_scaler.fit(y_train)

    def scale_seq(values):
        shape = values.shape
        return seq_scaler.transform(values.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

    x_seq_train_scaled = scale_seq(x_seq_train)
    x_seq_val_scaled = scale_seq(x_seq_val)
    x_seq_test_scaled = scale_seq(x_seq_test)
    x_tab_train_scaled = tab_scaler.transform(x_tab_train).astype(np.float32)
    x_tab_val_scaled = tab_scaler.transform(x_tab_val).astype(np.float32)
    x_tab_test_scaled = tab_scaler.transform(x_tab_test).astype(np.float32)
    y_train_scaled = target_scaler.transform(y_train).astype(np.float32)
    y_val_scaled = target_scaler.transform(y_val).astype(np.float32)
    y_test_scaled = target_scaler.transform(y_test).astype(np.float32)

    train_loader = torch.utils.data.DataLoader(
        DiseaseDataset(x_seq_train_scaled, x_tab_train_scaled, y_train_scaled),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        DiseaseDataset(x_seq_val_scaled, x_tab_val_scaled, y_val_scaled),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        DiseaseDataset(x_seq_test_scaled, x_tab_test_scaled, y_test_scaled),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
    )

    model = DiseasePatchTSTFusion(
        seq_len=CONFIG["seq_len"],
        seq_feature_dim=seq_feature_dim,
        tab_feature_dim=tab_feature_dim,
        output_dim=len(target_cols),
        patch_len=CONFIG["patch_len"],
        stride=CONFIG["stride"],
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        e_layers=CONFIG["e_layers"],
        d_ff=CONFIG["d_ff"],
        dropout=CONFIG["dropout"],
    ).to(device)

    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        min_lr=1e-5,
    )

    best_val_loss = float("inf")
    bad_epochs = 0
    best_model_path = os.path.join(output_dir, "patchtst_best.pth")

    print(f"\n========== Training {disease_name} ==========")
    print(json.dumps(split_info, ensure_ascii=False, indent=2))

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"{disease_name} | Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"{disease_name}: saved best PatchTST model.")
        else:
            bad_epochs += 1

        if bad_epochs >= CONFIG["patience"]:
            print(f"{disease_name}: early stopping.")
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, pred_scaled, true_scaled = evaluate(model, test_loader, criterion, device)

    patchtst_delta = target_scaler.inverse_transform(pred_scaled)
    true_delta = target_scaler.inverse_transform(true_scaled)
    patchtst_pred = np.clip(prev_y_test + patchtst_delta, 0, 100)
    true_current = y_current_test

    mean_delta = np.repeat(y_train.mean(axis=0, keepdims=True), len(true_current), axis=0)
    mean_delta_pred = np.clip(prev_y_test + mean_delta, 0, 100)
    prev_value_pred = np.clip(prev_y_test, 0, 100)

    reports = {
        "patchtst": metric_report(true_current, patchtst_pred, target_cols),
        "train_mean_delta_baseline": metric_report(true_current, mean_delta_pred, target_cols),
        "previous_value_baseline": metric_report(true_current, prev_value_pred, target_cols),
    }

    print(f"{disease_name} Test Loss:", test_loss)
    print(json.dumps(reports, ensure_ascii=False, indent=2))

    result_df = pd.DataFrame()
    for col in ["site_id", "site_name", "varieties", "disease_series_id", "date_str", "date"]:
        if col in work_df.columns:
            result_df[col] = work_df.loc[test_mask, col].values

    for i, col in enumerate(target_cols):
        result_df[f"prev_{col}"] = prev_y_test[:, i]
        result_df[f"true_{col}"] = true_current[:, i]
        result_df[f"patchtst_pred_{col}"] = patchtst_pred[:, i]
        result_df[f"mean_delta_pred_{col}"] = mean_delta_pred[:, i]
        result_df[f"prev_value_pred_{col}"] = prev_value_pred[:, i]
        result_df[f"true_delta_{col}"] = true_delta[:, i]
        result_df[f"patchtst_delta_{col}"] = patchtst_delta[:, i]

    result_path = os.path.join(output_dir, "test_predictions.xlsx")
    result_df.to_excel(result_path, index=False)

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                **CONFIG,
                "disease_name": disease_name,
                "seq_col": SEQ_COL,
                "target_cols": target_cols,
                "prev_target_cols": prev_cols,
                "delta_target_cols": delta_cols,
                "tab_feature_cols": tab_feature_cols,
                "seq_feature_dim": int(seq_feature_dim),
                "tab_feature_dim": int(tab_feature_dim),
                "split_info": split_info,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(os.path.join(output_dir, "seq_scaler.pkl"), "wb") as f:
        pickle.dump(seq_scaler, f)
    with open(os.path.join(output_dir, "tab_scaler.pkl"), "wb") as f:
        pickle.dump(tab_scaler, f)
    with open(os.path.join(output_dir, "target_scaler.pkl"), "wb") as f:
        pickle.dump(target_scaler, f)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_loss": float(test_loss),
                "reports": reports,
                "split_info": split_info,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "disease_name": disease_name,
        "test_loss": float(test_loss),
        "reports": reports,
        "split_info": split_info,
        "prediction_file": result_path,
    }


def main():
    set_seed(CONFIG["seed"])
    device = torch.device("cpu")

    df = pd.read_csv(DATA_PATH)
    validate_required_columns(df)

    print("Using device:", device)
    print("Rows:", len(df))
    print("Sites:", df["site_id"].nunique())
    if "disease_series_id" in df.columns:
        print("Disease series:", df["disease_series_id"].nunique())

    all_results = {}
    for disease_name, target_cols in DISEASE_GROUPS.items():
        all_results[disease_name] = train_disease_model(
            df=df,
            disease_name=disease_name,
            target_cols=target_cols,
            device=device,
        )

    summary_path = os.path.join(OUTPUT_DIR, "all_disease_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\nAll disease metrics saved:", summary_path)


if __name__ == "__main__":
    main()
