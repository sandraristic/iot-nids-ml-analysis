"""
ToN-IoT Train_Test_Network — Poglavlje 6.2: Priprema podataka
Korak 2: Feature engineering, enkodiranje kategoričkih obeležja,
podela na trening/test skup i normalizacija numeričkih obeležja.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

IN_PATH = "thesis_data/step1_cleaned.csv"
df = pd.read_csv(IN_PATH)
print(f"Ulazni oblik: {df.shape}")

# --- 1. Feature engineering pre enkodiranja ---
# src_port je efemeran (nasumično biran od strane klijenta) i ne nosi
# stabilan signal za klasifikaciju, pa se uklanja da bi se smanjio šum.
# dns_query je tekstualno polje vrlo visoke kardinalnosti (14.149 jedinstvenih
# vrednosti) — umesto potpunog uklanjanja, izvodimo binarno obeležje
# has_dns_query koje zadržava informaciju o tome da li je DNS upit uopšte postojao.
df["has_dns_query"] = (df["dns_query"] != "-").astype(int)
df = df.drop(columns=["src_port", "dns_query"])

# --- 2. Odvajanje labela (binarna 'label' i višeklasna 'type') ---
y_binary = df["label"]
y_multiclass = df["type"]
X = df.drop(columns=["label", "type"])

# --- 3. Enkodiranje kategoričkih obeležja (one-hot) ---
categorical_cols = X.select_dtypes(include="object").columns.tolist()
# dns_qclass, dns_qtype, dns_rcode su numerički kodovi sa malim brojem
# jedinstvenih vrednosti — tretiramo ih kao kategoričke radi ispravnog enkodiranja
low_card_numeric = ["dns_qclass", "dns_qtype", "dns_rcode"]
categorical_cols += low_card_numeric

print(f"\nKategorička obeležja za one-hot enkodiranje ({len(categorical_cols)}): {categorical_cols}")

X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
print(f"Oblik nakon one-hot enkodiranja: {X_encoded.shape}")

# --- 4. Podela na trening i test skup (80/20), stratifikovano po tipu napada ---
X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = train_test_split(
    X_encoded, y_binary, y_multiclass,
    test_size=0.20,
    random_state=42,
    stratify=y_multiclass,
)
print(f"\nTrening skup: {X_train.shape}, Test skup: {X_test.shape}")

# --- 5. Normalizacija numeričkih obeležja (StandardScaler, fit SAMO na treningu) ---
numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
# isključujemo binarne/one-hot kolone (0/1) iz standardizacije — normalizuju se
# samo "prave" numeričke veličine (bajtovi, paketi, trajanje...)
numeric_cols = [c for c in numeric_cols if X_train[c].nunique() > 2]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

print(f"\nBroj standardizovanih numeričkih obeležja: {len(numeric_cols)}")
print(f"Primeri: {numeric_cols[:8]}")

# --- 6. Čuvanje rezultata ---
X_train_scaled.to_csv("thesis_data/X_train.csv", index=False)
X_test_scaled.to_csv("thesis_data/X_test.csv", index=False)
y_train_bin.to_csv("thesis_data/y_train_binary.csv", index=False)
y_test_bin.to_csv("thesis_data/y_test_binary.csv", index=False)
y_train_multi.to_csv("thesis_data/y_train_multiclass.csv", index=False)
y_test_multi.to_csv("thesis_data/y_test_multiclass.csv", index=False)

print("\nSvi fajlovi sačuvani u thesis_data/")
print(f"\nFinalni broj obeležja (nakon enkodiranja): {X_train_scaled.shape[1]}")

# Distribucija klasa u train/test (provera da je stratifikacija uspela)
print("\n=== Distribucija tipova napada u trening skupu ===")
print(y_train_multi.value_counts())
print("\n=== Distribucija tipova napada u test skupu ===")
print(y_test_multi.value_counts())
