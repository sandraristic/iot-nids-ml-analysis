import pandas as pd

DATA_DIR = "thesis_data"

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train_binary.csv").squeeze()

# Pirsonova korelacija svakog obeležja sa binarnom oznakom label
corr = X_train.corrwith(y_train).abs().sort_values(ascending=False)

print("=== Pet obeležja sa najvećom apsolutnom korelacijom ===")
print(corr.head(5))

# Provera znaka korelacije za dns_rejected_F (za tumačenje u Tabeli 12)
print(f"\nZnak korelacije dns_rejected_F: {X_train['dns_rejected_F'].corr(y_train):.4f}")

n_weak = (corr < 0.01).sum()
print(f"\nBroj obeležja sa |korelacijom| < 0,01: {n_weak} od {len(corr)}")

corr.to_csv(f"{DATA_DIR}/feature_correlations.csv", header=["abs_correlation"])
print(f"\nSve korelacije sačuvane u {DATA_DIR}/feature_correlations.csv")