import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.ticker import FuncFormatter

DATA_DIR = "thesis_data"
OUT_DIR = "thesis_data/images"

import os
os.makedirs(OUT_DIR, exist_ok=True)

# Уједначен стил за све графиконе у раду
plt.rcParams.update({
    "font.family": "DejaVu Sans",   # једини стандардни фонт са пуном ћириличном подршком
    "font.size": 10,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#44545f",
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def sacuvaj(fig, ime):
    """Чува исти графикон у PDF (за штампу) и PNG (за Word/Docs)."""
    fig.savefig(f"{OUT_DIR}/{ime}.pdf")
    fig.savefig(f"{OUT_DIR}/{ime}.png")
    plt.close(fig)
    print(f"  сачувано: {OUT_DIR}/{ime}.pdf i .png")


# ==========================================================
# Учитавање података
# ==========================================================
print("Учитавање података...")
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train_binary.csv").squeeze()
y_test = pd.read_csv(f"{DATA_DIR}/y_test_binary.csv").squeeze()
y_train_multi = pd.read_csv(f"{DATA_DIR}/y_train_multiclass.csv").squeeze()
y_test_multi = pd.read_csv(f"{DATA_DIR}/y_test_multiclass.csv").squeeze()
print(f"  трениг: {X_train.shape}, тест: {X_test.shape}")


# ==========================================================
# СЛИКА 1 — ROC криве сва четири модела (поглавље 7.1)
# ==========================================================
print("\n[1/3] Тренирање модела ради конструкције ROC кривих...")

krive = []   # (назив, fpr, tpr, auc, стил)

# --- Random Forest ---
print("  Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=1)
rf.fit(X_train, y_train)
proba_rf = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, proba_rf)
krive.append(("Random Forest", fpr, tpr, roc_auc_score(y_test, proba_rf), "-", "#1b4f72", 2.2))

# --- XGBoost ---
print("  XGBoost...")
xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=1, eval_metric="logloss")
xgb.fit(X_train, y_train)
proba_xgb = xgb.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, proba_xgb)
krive.append(("XGBoost", fpr, tpr, roc_auc_score(y_test, proba_xgb), "--", "#2e86a1", 2.0))

# --- SVM (идентичан узорак као у Прилогу В) ---
SVM_SAMPLE_SIZE = 15000
print(f"  SVM (узорак {SVM_SAMPLE_SIZE:,})...".replace(",", "."))
rng = np.random.RandomState(42)
idx_0 = y_train[y_train == 0].index
idx_1 = y_train[y_train == 1].index
n0 = int(SVM_SAMPLE_SIZE * (len(idx_0) / len(y_train)))
n1 = SVM_SAMPLE_SIZE - n0
sample_idx = np.concatenate([
    rng.choice(idx_0, size=n0, replace=False),
    rng.choice(idx_1, size=n1, replace=False),
])
svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
svm.fit(X_train.loc[sample_idx], y_train.loc[sample_idx])
proba_svm = svm.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, proba_svm)
krive.append(("SVM (узорак 15k)", fpr, tpr, roc_auc_score(y_test, proba_svm), "-.", "#8e6c1f", 1.8))

# --- Isolation Forest (идентична поставка као у Прилогу Г) ---
print("  Isolation Forest...")
X_train_normal = X_train[y_train == 0]
iso = IsolationForest(n_estimators=100, contamination=float(y_test.mean()),
                      random_state=42, n_jobs=1)
iso.fit(X_train_normal)
skor_iso = -iso.score_samples(X_test)
fpr, tpr, _ = roc_curve(y_test, skor_iso)
krive.append(("Isolation Forest", fpr, tpr, roc_auc_score(y_test, skor_iso), ":", "#a63a3a", 2.0))

# --- Цртање ---
fig, ax = plt.subplots(figsize=(6.6, 5.4))

for naziv, fpr, tpr, auc, stil, boja, deb in krive:
    ax.plot(fpr, tpr, stil, color=boja, linewidth=deb,
            label=f"{naziv}  (AUC = {auc:.4f})".replace(".", ","))

ax.plot([0, 1], [0, 1], color="#9aa7b0", linewidth=1.0, linestyle=(0, (2, 3)),
        label="насумично погађање (AUC = 0,5)")

ax.set_xlim(-0.01, 1.0)
ax.set_ylim(0.0, 1.01)
zarez = FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
ax.xaxis.set_major_formatter(zarez)
ax.yaxis.set_major_formatter(zarez)
ax.set_xlabel("Стопа лажно позитивних (FPR)")
ax.set_ylabel("Стопа тачно позитивних (TPR)")
ax.set_title("ROC криве примењених модела на тест скупу", fontsize=12, pad=12)
ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9)
ax.grid(True, linewidth=0.4, color="#dde3e8")
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

sacuvaj(fig, "slika_roc_krive")

print("\n  Провера — AUC вредности треба да одговарају Табели 13:")
for naziv, _, _, auc, _, _, _ in krive:
    print(f"    {naziv:22s} AUC = {auc:.5f}")


# ==========================================================
# СЛИКА 2 — Дистрибуција класа (поглавље 6.2.8)
# ==========================================================
print("\n[2/3] Дистрибуција класа...")

raspodela = y_train_multi.value_counts().sort_values()
boje = ["#a63a3a" if k == "mitm" else "#4a7ea1" for k in raspodela.index]

fig, ax = plt.subplots(figsize=(7.0, 4.6))
pozicije = np.arange(len(raspodela))
ax.barh(pozicije, raspodela.values, color=boje, height=0.68, edgecolor="#2b3a44", linewidth=0.5)

ax.set_yticks(pozicije)
ax.set_yticklabels(raspodela.index)
ax.set_xscale("log")
ax.set_xlabel("Број инстанци у тренинг скупу (логаритамска скала)")
ax.set_title("Дистрибуција класа у тренинг скупу", fontsize=12, pad=12)

for i, v in enumerate(raspodela.values):
    ax.text(v * 1.15, i, f"{v:,}".replace(",", "."), va="center", fontsize=9, color="#2b3a44")

ax.set_xlim(right=raspodela.max() * 3.2)
ax.grid(True, axis="x", linewidth=0.4, color="#dde3e8")
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

odnos = raspodela.max() / raspodela.min()
ax.text(0.98, 0.04, f"однос највеће и најмање класе: {odnos:.0f} : 1",
        transform=ax.transAxes, ha="right", fontsize=9.5, style="italic", color="#a63a3a")

sacuvaj(fig, "slika_distribucija_klasa")
print(f"  најмања класа: {raspodela.index[0]} ({raspodela.iloc[0]} инстанци)")


# ==========================================================
# СЛИКА 3 — Важност обележја (поглавље 6.3.4)
# ==========================================================
print("\n[3/3] Важност обележја (Random Forest)...")

vaznost = pd.Series(rf.feature_importances_, index=X_train.columns) \
            .sort_values(ascending=False).head(12).sort_values()

fig, ax = plt.subplots(figsize=(7.0, 4.8))
ax.barh(np.arange(len(vaznost)), vaznost.values, color="#4a7ea1",
        height=0.68, edgecolor="#2b3a44", linewidth=0.5)
ax.set_yticks(np.arange(len(vaznost)))
ax.set_yticklabels(vaznost.index, fontsize=9)
ax.set_xlabel("Важност обележја (Gini)")
ax.set_title("Дванаест најзначајнијих обележја — Random Forest", fontsize=12, pad=12)

for i, v in enumerate(vaznost.values):
    ax.text(v + vaznost.max() * 0.015, i, f"{v:.3f}".replace(".", ","),
            va="center", fontsize=8.5, color="#2b3a44")

ax.set_xlim(right=vaznost.max() * 1.18)
ax.grid(True, axis="x", linewidth=0.4, color="#dde3e8")
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

sacuvaj(fig, "slika_vaznost_obelezja")

print(f"\nГотово. Сви графикони су у фолдеру: {OUT_DIR}/")
