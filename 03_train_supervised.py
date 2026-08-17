"""
ToN-IoT — Poglavlje 6.3: Primena algoritama mašinskog učenja
Korak 1: Nadgledani modeli — Random Forest, XGBoost, SVM (binarna klasifikacija: normal/napad)
"""
import pandas as pd
import numpy as np
import time
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

DATA_DIR = "thesis_data"
results = {}

print("Učitavanje podataka...")
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train_binary.csv").squeeze()
y_test = pd.read_csv(f"{DATA_DIR}/y_test_binary.csv").squeeze()
print(f"Trening: {X_train.shape}, Test: {X_test.shape}")

def evaluate(name, y_true, y_pred, y_proba, train_time):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Vreme treniranja: {train_time:.1f}s")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.6f}")
    print(f"Confusion matrix:\n{cm}")
    results[name] = {
        "train_time_sec": round(train_time, 1),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 6),
        "confusion_matrix": cm.tolist(),
        "n_train": len(y_true) if name == "dummy" else None,
    }
    return results[name]

# ============================================================
# 1. RANDOM FOREST (pun trening skup)
# ============================================================
print("\n" + "="*60)
print("Treniranje Random Forest...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=1)
rf.fit(X_train, y_train)
t_rf = time.time() - t0

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
evaluate("Random Forest", y_test, y_pred_rf, y_proba_rf, t_rf)

# feature importance - top 10
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nTop 10 najvažnijih obeležja (Random Forest):")
print(importances.head(10))
results["rf_feature_importance_top10"] = importances.head(10).to_dict()

# ============================================================
# 2. XGBOOST (pun trening skup)
# ============================================================
print("\n" + "="*60)
print("Treniranje XGBoost...")
t0 = time.time()
xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                     random_state=42, n_jobs=1, eval_metric='logloss')
xgb.fit(X_train, y_train)
t_xgb = time.time() - t0

y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
evaluate("XGBoost", y_test, y_pred_xgb, y_proba_xgb, t_xgb)

# ============================================================
# 3. SVM (na stratifikovanom uzorku, zbog O(n^2-n^3) slozenosti - vidi poglavlje 4.2)
# ============================================================
print("\n" + "="*60)
print("Treniranje SVM na uzorku (15.000 instanci, stratifikovano)...")
SVM_SAMPLE_SIZE = 15000
rng = np.random.RandomState(42)
idx_0 = y_train[y_train == 0].index
idx_1 = y_train[y_train == 1].index
n0 = int(SVM_SAMPLE_SIZE * (len(idx_0) / len(y_train)))
n1 = SVM_SAMPLE_SIZE - n0
sample_idx = np.concatenate([
    rng.choice(idx_0, size=n0, replace=False),
    rng.choice(idx_1, size=n1, replace=False),
])
X_train_svm = X_train.loc[sample_idx]
y_train_svm = y_train.loc[sample_idx]
print(f"SVM trening uzorak: {X_train_svm.shape}, distribucija: {y_train_svm.value_counts().to_dict()}")

t0 = time.time()
svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
svm.fit(X_train_svm, y_train_svm)
t_svm = time.time() - t0

y_pred_svm = svm.predict(X_test)
y_proba_svm = svm.predict_proba(X_test)[:, 1]
evaluate("SVM (uzorak 15k)", y_test, y_pred_svm, y_proba_svm, t_svm)

# ============================================================
# Sačuvaj sve rezultate
# ============================================================
with open(f"{DATA_DIR}/results_supervised.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n\nSvi rezultati sačuvani u {DATA_DIR}/results_supervised.json")
