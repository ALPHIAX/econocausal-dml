import pandas as pd

INPUT_PATH = "Data/Processed/dml_modeling_data.csv"
OUTPUT_PATH = "Data/Processed/qini_data.csv"

df = pd.read_csv(INPUT_PATH)

# Low-price treatment proxy:
# Bottom 25% of observed prices are treated as relatively low-price observations.
price_threshold = df["unit_price"].quantile(0.25)

df["treatment"] = (df["unit_price"] <= price_threshold).astype(int)

qini_df = df[
    ["unit_price", "quantity", "treatment"]
].copy()

qini_df.to_csv(OUTPUT_PATH, index=False)

print("=== Qini Treatment Preparation ===")
print("Price threshold:", price_threshold)
print("Total observations:", len(qini_df))
print("Treated observations:", qini_df["treatment"].sum())
print("Control observations:", (qini_df["treatment"] == 0).sum())
print("Treatment rate:", qini_df["treatment"].mean())
print("Saved:", OUTPUT_PATH)

