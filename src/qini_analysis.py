import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

QINI_DATA_PATH = "Data/Processed/qini_data.csv"
ITE_DATA_PATH = "Data/Processed/dml_ite_results.csv"
OUTPUT_PATH = "Data/Processed/qini_curve.html"

qini_df = pd.read_csv(QINI_DATA_PATH)
ite_df = pd.read_csv(ITE_DATA_PATH)

# --------------------------------------------------
# 1. Reproduce the DML test split
# --------------------------------------------------

indices = np.arange(len(qini_df))

_, test_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=42
)

qini_test = qini_df.iloc[test_indices].copy().reset_index(drop=True)

# --------------------------------------------------
# 2. Validate ITE alignment
# --------------------------------------------------

if len(qini_test) != len(ite_df):
    raise ValueError(
        f"Test set ({len(qini_test)}) and ITE results "
        f"({len(ite_df)}) have different row counts."
    )

qini_test["estimated_ite"] = ite_df["estimated_ite"].values

print("=== Qini Input Validation ===")
print("Total modeling observations:", len(qini_df))
print("DML test observations:", len(qini_test))
print("ITE observations:", len(ite_df))

# --------------------------------------------------
# 3. Rank observations by estimated treatment effect
# --------------------------------------------------

qini_test = qini_test.sort_values(
    "estimated_ite",
    ascending=False
).reset_index(drop=True)

qini_test["rank"] = np.arange(1, len(qini_test) + 1)

# --------------------------------------------------
# 4. Calculate cumulative treatment/control outcomes
# --------------------------------------------------

qini_test["treated_outcome"] = (
    qini_test["treatment"] * qini_test["quantity"]
)

qini_test["control_outcome"] = (
    (1 - qini_test["treatment"]) * qini_test["quantity"]
)

qini_test["cumulative_treated_outcome"] = (
    qini_test["treated_outcome"].cumsum()
)

qini_test["cumulative_control_outcome"] = (
    qini_test["control_outcome"].cumsum()
)

qini_test["cumulative_treated"] = (
    qini_test["treatment"].cumsum()
)

qini_test["cumulative_control"] = (
    (1 - qini_test["treatment"]).cumsum()
)

# --------------------------------------------------
# 5. Calculate Qini gain
# --------------------------------------------------

qini_test["qini_gain"] = (
    qini_test["cumulative_treated_outcome"]
    - qini_test["cumulative_control_outcome"]
    * (
        qini_test["cumulative_treated"]
        / qini_test["cumulative_control"].replace(0, np.nan)
    )
)

qini_test["population_fraction"] = (
    qini_test["rank"] / len(qini_test)
)

# --------------------------------------------------
# 6. Create Plotly Qini curve
# --------------------------------------------------

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=qini_test["population_fraction"],
        y=qini_test["qini_gain"],
        mode="lines",
        name="DML estimated ITE"
    )
)

fig.add_hline(
    y=0,
    line_dash="dash",
    name="Baseline"
)

fig.update_layout(
    title="Qini Curve — Low-Price Treatment Proxy",
    xaxis_title="Population Fraction Targeted",
    yaxis_title="Cumulative Qini Gain",
    template="plotly_white"
)

fig.write_html(OUTPUT_PATH)

# --------------------------------------------------
# 7. Summary
# --------------------------------------------------

max_gain_idx = qini_test["qini_gain"].idxmax()

print("\n=== Qini Analysis ===")
print("Treatment: Low-price treatment proxy")
print("Treatment threshold: unit_price <= 1.25")
print("Observations:", len(qini_test))
print(
    "Maximum Qini gain:",
    qini_test.loc[max_gain_idx, "qini_gain"]
)
print(
    "Population fraction at maximum gain:",
    qini_test.loc[max_gain_idx, "population_fraction"]
)
print("Saved:", OUTPUT_PATH)

print("\n=== Qini Gain at Population Fractions ===")

for fraction in [0.10, 0.20, 0.30, 0.50, 0.75, 1.00]:
    index = int(len(qini_test) * fraction) - 1
    print(
        f"{fraction:.0%}: "
        f"{qini_test.loc[index, 'qini_gain']:.2f}"
    ) 