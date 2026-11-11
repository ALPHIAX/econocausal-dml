from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
data = root / "Data" / "Processed"

base = pd.read_csv(data / "dml_ite_results.csv")
alt = pd.read_csv(data / "dml_ite_results_alpha10.csv")

assert len(base) == len(alt), "Prediction row counts differ."

for col in ["invoice_id", "stock_code", "country"]:
    assert np.array_equal(
        base[col].astype(str).to_numpy(),
        alt[col].astype(str).to_numpy()
    ), f"Observation alignment failed: {col}"

for col in ["unit_price", "quantity"]:
    assert np.allclose(base[col], alt[col]), f"Mismatch: {col}"

cate = base["estimated_cate"].to_numpy()
alt_cate = alt["estimated_cate"].to_numpy()
price = base["unit_price"].to_numpy()
quantity = base["quantity"].to_numpy()

assert all(np.isfinite(x).all() for x in [cate, alt_cate, price, quantity])
assert (price > 0).all()

rows = []
for pct in [0.05, 0.10, 0.20]:
    delta_price = price * pct
    delta_quantity = cate * delta_price

    rows.append({
        "price_increase_pct": pct * 100,
        "observations": len(base),
        "median_price_change": np.median(delta_price),
        "mean_predicted_quantity_change": np.mean(delta_quantity),
        "median_predicted_quantity_change": np.median(delta_quantity),
        "p05_quantity_change": np.quantile(delta_quantity, 0.05),
        "p95_quantity_change": np.quantile(delta_quantity, 0.95),
        "pct_reduction_exceeds_observed_quantity":
            np.mean(delta_quantity < -quantity) * 100,
        "pct_negative_implied_quantity":
            np.mean(quantity + delta_quantity < 0) * 100
    })

pd.DataFrame(rows).to_csv(
    data / "cate_magnitude_price_shock_sensitivity.csv", index=False
)

comparison = pd.DataFrame([{
    "observations": len(base),
    "baseline_mean_cate": np.mean(cate),
    "alternative_mean_cate": np.mean(alt_cate),
    "baseline_median_cate": np.median(cate),
    "alternative_median_cate": np.median(alt_cate),
    "prediction_correlation": np.corrcoef(cate, alt_cate)[0, 1],
    "sign_agreement_pct": np.mean(np.sign(cate) == np.sign(alt_cate)) * 100,
    "mean_absolute_difference": np.mean(np.abs(cate - alt_cate))
}])

comparison.to_csv(
    data / "cate_magnitude_model_comparison.csv", index=False
)

print("CATE magnitude diagnostic completed.")
print(pd.DataFrame(rows).to_string(index=False))
print("Saved both diagnostic CSV files.")
