"""
ToN-IoT Train_Test_Network — Poglavlje 6.2: Priprema podataka
Korak 1: Čišćenje podataka i uklanjanje kolona sa slabim informativnim sadržajem
"""
import pandas as pd
import numpy as np

RAW_PATH = "thesis_data/TON_IoT_Train_Test_Network.csv"
OUT_PATH = "thesis_data/step1_cleaned.csv"

df = pd.read_csv(RAW_PATH)
print(f"Originalni oblik: {df.shape}")

# --- 1. Uklanjanje potpunih duplikata ---
n_before = len(df)
df = df.drop_duplicates()
n_after = len(df)
print(f"Uklonjeno duplikata: {n_before - n_after} ({(n_before-n_after)/n_before*100:.2f}%)")

# --- 2. Identifikacija kolona sa vrlo visokim procentom '-' (Zeek placeholder) ---
str_cols = df.select_dtypes(include="object").columns
placeholder_pct = {}
for c in str_cols:
    pct = (df[c] == "-").mean() * 100
    if pct > 0:
        placeholder_pct[c] = pct

# Kolone koje uklanjamo: >95% placeholder vrednosti = premalo signala za modele
DROP_HIGH_MISSING = [c for c, p in placeholder_pct.items() if p > 95]
print(f"\nKolone uklonjene zbog >95% '-' vrednosti ({len(DROP_HIGH_MISSING)}):")
for c in DROP_HIGH_MISSING:
    print(f"  {c}: {placeholder_pct[c]:.1f}% placeholder")

# --- 3. Uklanjanje identifikacionih kolona (curenje labele / visoka kardinalnost) ---
# src_ip/dst_ip su direktno korišćeni za generisanje labele (Kali Linux IP opseg),
# pa bi njihovo zadržavanje predstavljalo curenje informacije o cilju (label leakage).
# ts (timestamp) i src_port/dst_port kao sirovi identifikatori takođe se uklanjaju
# iz skupa obeležja (port se može po potrebi binovati kasnije, van osnovnog čišćenja).
DROP_IDENTIFIERS = ["ts", "src_ip", "dst_ip"]

drop_cols = list(set(DROP_HIGH_MISSING + DROP_IDENTIFIERS))
df_clean = df.drop(columns=drop_cols)

print(f"\nUkupno uklonjeno kolona: {len(drop_cols)}")
print(f"Preostalo kolona: {df_clean.shape[1]}")
print(f"Oblik nakon 1. koraka čišćenja: {df_clean.shape}")

df_clean.to_csv(OUT_PATH, index=False)
print(f"\nSačuvano u: {OUT_PATH}")

# Sačuvaj i listu izbačenih kolona za dokumentaciju u radu
with open("thesis_data/scripts/dropped_columns.txt", "w") as f:
    f.write("Kolone izbačene zbog >95% placeholder ('-') vrednosti:\n")
    for c in DROP_HIGH_MISSING:
        f.write(f"  {c}: {placeholder_pct[c]:.1f}%\n")
    f.write("\nKolone izbačene kao identifikatori (rizik od label leakage / visoka kardinalnost):\n")
    for c in DROP_IDENTIFIERS:
        f.write(f"  {c}\n")
