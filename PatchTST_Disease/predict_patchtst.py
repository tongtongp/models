import ast
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch

from patchtst_model import DiseasePatchTSTFusion


PROJECT_DIR = r"C:\Users\tong2\Desktop\models\PatchTST_Disease"
DATA_PATH = os.path.join(PROJECT_DIR, "data", "corn_disease.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
DISEASE_NAMES = ["gray", "blight", "white"]


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
        raise ValueError(f"Cannot parse weather sequence: {type(value)}")

    if arr.ndim != 2:
        raise ValueError(f"weather_seq_28 must be 2D, got shape={arr.shape}")

    return arr.astype(np.float32)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_one_disease(row, disease_dir, device):
    config_path = os.path.join(disease_dir, "config.json")
    model_path = os.path.join(disease_dir, "patchtst_best.pth")
    seq_scaler_path = os.path.join(disease_dir, "seq_scaler.pkl")
    tab_scaler_path = os.path.join(disease_dir, "tab_scaler.pkl")
    target_scaler_path = os.path.join(disease_dir, "target_scaler.pkl")

    required_files = [
        config_path,
        model_path,
        seq_scaler_path,
        tab_scaler_path,
        target_scaler_path,
    ]
    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing model files:\n"
            + "\n".join(missing)
            + "\nRun train_patchtst.py first."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    seq_scaler = load_pickle(seq_scaler_path)
    tab_scaler = load_pickle(tab_scaler_path)
    target_scaler = load_pickle(target_scaler_path)

    seq_col = config["seq_col"]
    tab_feature_cols = config["tab_feature_cols"]
    target_cols = config["target_cols"]
    prev_target_cols = config["prev_target_cols"]

    for col in [seq_col, *tab_feature_cols, *prev_target_cols]:
        if col not in row.index:
            raise ValueError(f"Input row is missing column: {col}")

    x_seq_raw = parse_weather_seq(row[seq_col])
    x_tab_raw = row[tab_feature_cols].astype(float).values.astype(np.float32)

    x_seq_scaled = seq_scaler.transform(
        x_seq_raw.reshape(-1, x_seq_raw.shape[-1])
    ).reshape(x_seq_raw.shape).astype(np.float32)
    x_tab_scaled = tab_scaler.transform(x_tab_raw.reshape(1, -1)).astype(np.float32)

    x_seq = torch.tensor(x_seq_scaled[None, :, :], dtype=torch.float32).to(device)
    x_tab = torch.tensor(x_tab_scaled, dtype=torch.float32).to(device)

    model = DiseasePatchTSTFusion(
        seq_len=config["seq_len"],
        seq_feature_dim=config["seq_feature_dim"],
        tab_feature_dim=config["tab_feature_dim"],
        output_dim=len(target_cols),
        patch_len=config["patch_len"],
        stride=config["stride"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        e_layers=config["e_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        pred_scaled = model(x_seq, x_tab).cpu().numpy()

    pred_delta = target_scaler.inverse_transform(pred_scaled)[0]
    prev_targets = row[prev_target_cols].astype(float).values.astype(np.float32)
    pred = np.clip(prev_targets + pred_delta, 0, 100)

    result = {}
    for i, col in enumerate(target_cols):
        result[f"prev_{col}"] = float(prev_targets[i])
        result[f"pred_delta_{col}"] = float(pred_delta[i])
        result[f"pred_{col}"] = float(pred[i])

    return result


def main():
    device = torch.device("cpu")
    df = pd.read_csv(DATA_PATH)

    row = df.iloc[-1]
    result = {}

    for col in ["site_id", "site_name", "varieties", "disease_series_id", "date_str", "date"]:
        if col in df.columns:
            result[col] = row[col]

    for disease_name in DISEASE_NAMES:
        disease_dir = os.path.join(OUTPUT_DIR, disease_name)
        result.update(predict_one_disease(row, disease_dir, device))

    result_df = pd.DataFrame([result])
    output_file = os.path.join(OUTPUT_DIR, "latest_prediction.xlsx")
    result_df.to_excel(output_file, index=False)

    print("Prediction result:")
    print(result_df)
    print("Saved:", output_file)


if __name__ == "__main__":
    main()
