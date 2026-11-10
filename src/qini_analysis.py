"""Create an exploratory ranking profile of held-out DML CATE estimates.

This is NOT a Qini curve and does NOT measure causal uplift, policy value,
incremental outcomes, or treatment-ranking accuracy. It visualizes the model's
own predicted CATE distribution after sorting observations by predicted CATE.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "Data" / "Processed" / "cate_ranking_data.csv"
OUTPUT_PATH = ROOT / "Data" / "Processed" / "cate_ranking_diagnostic.html"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Run src/prepare_qini_data.py first. Missing: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)
    if "estimated_cate" not in df.columns:
        raise ValueError("Input must contain 'estimated_cate'.")

    cate = pd.to_numeric(df["estimated_cate"], errors="coerce")
    cate = cate.replace([np.inf, -np.inf], np.nan).dropna()

    if cate.empty:
        raise ValueError("No finite CATE estimates are available.")

    ranked = pd.DataFrame({"estimated_cate": cate})
    ranked = ranked.sort_values("estimated_cate", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["population_fraction"] = ranked["rank"] / len(ranked)
    ranked["cumulative_mean_cate"] = (
        ranked["estimated_cate"].expanding().mean()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ranked["population_fraction"],
            y=ranked["cumulative_mean_cate"],
            mode="lines",
            name="Cumulative mean predicted CATE",
            hovertemplate=(
                "Population ranked: %{x:.1%}<br>"
                "Cumulative mean predicted CATE: %{y:.4f}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_dash="dash", name="Zero predicted effect")
    fig.update_layout(
        title="Exploratory CATE Ranking Profile (Not a Qini/Uplift Curve)",
        xaxis_title="Fraction of held-out observations, ranked by predicted CATE",
        yaxis_title="Cumulative mean predicted CATE",
        template="plotly_white",
        annotations=[
            dict(
                text=(
                    "Descriptive model-output diagnostic only. "
                    "Does not validate causal uplift or policy value."
                ),
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.12,
                showarrow=False,
            )
        ],
    )
    fig.write_html(OUTPUT_PATH)

    print("=== CATE Ranking Diagnostic ===")
    print("Observations:", len(ranked))
    print("Mean predicted CATE:", round(ranked["estimated_cate"].mean(), 6))
    print("Median predicted CATE:", round(ranked["estimated_cate"].median(), 6))
    print("Positive predicted CATE:", int((ranked["estimated_cate"] > 0).sum()))
    print("Negative predicted CATE:", int((ranked["estimated_cate"] < 0).sum()))
    print("Saved:", OUTPUT_PATH)
    print(
        "LIMITATION: this ranks the model's own predictions; "
        "it is not an out-of-sample causal uplift evaluation."
    )


if __name__ == "__main__":
    main()