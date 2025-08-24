from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# --- paths ---
project_path = Path(__file__).resolve().parent
data_path = project_path / "data" / "census.csv"
model_dir = project_path / "model"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "model.pkl"
encoder_path = model_dir / "encoder.pkl"
lb_path = model_dir / "label_binarizer.pkl"
slice_output_path = project_path / "slice_output.txt"

print(data_path)
data = pd.read_csv(data_path)

# --- split ---
train, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["salary"]
)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# --- train ---
model = train_model(X_train, y_train)

# --- save artifacts ---
save_model(model, model_path)
save_model(encoder, encoder_path)
save_model(lb, lb_path)

# --- (optional) reload to prove it works ---
model = load_model(model_path)

# --- inference & metrics ---
preds = inference(model, X_test)
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# --- slice metrics ---
# start fresh each run
slice_output_path.write_text("", encoding="utf-8")
with slice_output_path.open("a", encoding="utf-8") as f:
    for col in cat_features:
        for slicevalue in sorted(v for v in test[col].dropna().unique()):
            count = (test[col] == slicevalue).sum()
            # CALL POSITIONALLY: (model, data, feature, value, cat_features, label, encoder, lb)
            p_s, r_s, fb_s = performance_on_categorical_slice(
                model, test, col, slicevalue, cat_features, "salary", encoder, lb
            )
            print(f"Precision: {p_s:.4f} | Recall: {r_s:.4f} | F1: {fb_s:.4f}", file=f)
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)