import pandas as pd

DATA_PATH = "Data/Processed/cleaned_online_retail_dml.csv"

df = pd.read_csv(DATA_PATH)

print("=== DML Dataset Validation ===")
print(f"Shape: {df.shape}")

print("\n=== Columns ===")
print(df.columns.tolist())

print("\n=== Missing Values ===")
print(df.isna().sum())

print("\n=== Numeric Summary ===")
print(
    df[["quantity", "unit_price", "customer_id", "total_amount"]].describe()
)

print("\n=== Basic Checks ===")
print(f"Non-positive quantity: {(df['quantity'] <= 0).sum()}")
print(f"Non-positive unit price: {(df['unit_price'] <= 0).sum()}")

print("\nValidation completed.")