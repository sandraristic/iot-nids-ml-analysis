"""
ToN-IoT — Poglavlje 6.3: Primena algoritama mašinskog učenja
Korak 3: Provera efekta balansiranja klasa (class_weight) na najmanju klasu (MITM),
višeklasna klasifikacija (Random Forest).
"""
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score

DATA_DIR = "thesis_data"

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train_multi = pd.read_csv(f"{DATA_DIR}/y_train_multiclass.csv").squeeze()
y_test_multi = pd.read_csv(f"{DATA_DIR}/y_test_multiclass.csv").squeeze()

print(f"Trening: {X_train.shape}, klase: {y_train_multi.nunique()}")
print(y_train_multi.value_counts())

# --- Model 1: Random Forest BEZ balansiranja klasa ---
print("\n" + "="*60)
print("Random Forest (multiklasa) - BEZ balansiranja...")
rf_base = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=1)
rf_base.fit(X_train, y_train_multi)
y_pred_base = rf_base.predict(X_test)

report_base = classification_report(y_test_multi, y_pred_base, output_dict=True, zero_division=0)
mitm_recall_base = report_base.get("mitm", {}).get("recall", None)
macro_f1_base = report_base["macro avg"]["f1-score"]
print(f"MITM recall (bez balansiranja): {mitm_recall_base}")
print(f"Macro F1 (bez balansiranja): {macro_f1_base:.4f}")

# --- Model 2: Random Forest SA class_weight='balanced' ---
print("\n" + "="*60)
print("Random Forest (multiklasa) - SA class_weight='balanced'...")
rf_balanced = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42,
                                       n_jobs=1, class_weight="balanced")
rf_balanced.fit(X_train, y_train_multi)
y_pred_balanced = rf_balanced.predict(X_test)

report_balanced = classification_report(y_test_multi, y_pred_balanced, output_dict=True, zero_division=0)
mitm_recall_balanced = report_balanced.get("mitm", {}).get("recall", None)
macro_f1_balanced = report_balanced["macro avg"]["f1-score"]
print(f"MITM recall (sa balansiranjem): {mitm_recall_balanced}")
print(f"Macro F1 (sa balansiranjem): {macro_f1_balanced:.4f}")

print("\n" + "="*60)
print("=== POREĐENJE PO KLASAMA (recall) ===")
comparison = pd.DataFrame({
    "bez_balansiranja": {k: v["recall"] for k, v in report_base.items() if k in y_train_multi.unique()},
    "sa_balansiranjem": {k: v["recall"] for k, v in report_balanced.items() if k in y_train_multi.unique()},
})
print(comparison.sort_values("bez_balansiranja"))

results = {
    "bez_balansiranja": {
        "macro_f1": round(macro_f1_base, 4),
        "mitm_recall": round(mitm_recall_base, 4) if mitm_recall_base else None,
        "per_class_recall": {k: round(v["recall"], 4) for k, v in report_base.items() if k in y_train_multi.unique()},
    },
    "sa_balansiranjem": {
        "macro_f1": round(macro_f1_balanced, 4),
        "mitm_recall": round(mitm_recall_balanced, 4) if mitm_recall_balanced else None,
        "per_class_recall": {k: round(v["recall"], 4) for k, v in report_balanced.items() if k in y_train_multi.unique()},
    },
}

with open(f"{DATA_DIR}/results_balancing.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nRezultati sačuvani u {DATA_DIR}/results_balancing.json")
