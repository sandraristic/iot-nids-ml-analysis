"""
ToN-IoT — Poglavlje 6.3: Primena algoritama mašinskog učenja
Korak 2: Nenadgledani model — Isolation Forest
"""
import pandas as pd
import numpy as np
import time
import json
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

DATA_DIR = "thesis_data"

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train_binary.csv").squeeze()
y_test = pd.read_csv(f"{DATA_DIR}/y_test_binary.csv").squeeze()

# Isolation Forest se obučava BEZ oznaka (unsupervised), ali radi realističnog
# scenarija - trening isključivo nad NORMALNIM saobraćajem (najčešći pristup u
# praksi za anomaly detection: model uči profil normalnog ponašanja).
X_train_normal = X_train[y_train == 0]
print(f"Trening skup (samo normalan saobraćaj): {X_train_normal.shape}")

# Očekivani udeo anomalija u test skupu (koristi se kao 'contamination' parametar
# - u realnom scenariju ovo bi bila procena na osnovu domenskog znanja, ovde
# koristimo stvaran udeo napada iz test skupa radi poštene evaluacije)
contamination = y_test.mean()
print(f"Procenjeni udeo anomalija (contamination): {contamination:.4f}")

t0 = time.time()
iso = IsolationForest(
    n_estimators=100,
    contamination=contamination,
    random_state=42,
    n_jobs=1,
)
iso.fit(X_train_normal)
t_iso = time.time() - t0
print(f"Vreme treniranja: {t_iso:.1f}s")

# predict() vraća 1 (normalno) / -1 (anomalija) - mapiramo u 0/1 da odgovara label konvenciji
raw_pred = iso.predict(X_test)
y_pred_iso = np.where(raw_pred == -1, 1, 0)  # 1 = anomalija/napad

# score_samples: veći (manje negativan) = normalnije; invertujemo za ROC-AUC (veći skor = veća verovatnoća napada)
anomaly_score = -iso.score_samples(X_test)

acc = accuracy_score(y_test, y_pred_iso)
prec = precision_score(y_test, y_pred_iso)
rec = recall_score(y_test, y_pred_iso)
f1 = f1_score(y_test, y_pred_iso)
auc = roc_auc_score(y_test, anomaly_score)
cm = confusion_matrix(y_test, y_pred_iso)

print(f"\n=== Isolation Forest ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")
print(f"Confusion matrix:\n{cm}")

results = {
    "Isolation Forest": {
        "train_time_sec": round(t_iso, 1),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
        "contamination_param": round(float(contamination), 4),
        "n_train_normal_only": len(X_train_normal),
    }
}

with open(f"{DATA_DIR}/results_unsupervised.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nRezultati sačuvani u {DATA_DIR}/results_unsupervised.json")
