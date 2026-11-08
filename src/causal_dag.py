
"""
Provisional causal DAG for EconoCausal.

Research question:
How does product price affect transaction quantity,
and how does this effect vary across products and time?

Observed variable mapping:
- product: stock_code
- price: unit_price
- quantity: quantity
- time: temporal features derived from invoicedate
- country: country

DML design:
- Treatment (T): unit_price
- Outcome (Y): quantity
- Effect modifiers (X): product and temporal features
- Adjustment variables (W): country

All causal assumptions require validation.
"""

CAUSAL_GRAPH = """
digraph {
    product -> price;
    product -> quantity;

    time -> price;
    time -> quantity;

    country -> price;
    country -> quantity;

    price -> quantity;
}
"""


def get_causal_graph():
    return CAUSAL_GRAPH


if __name__ == "__main__":
    print("EconoCausal: Provisional Causal DAG")
    print("-----------------------------------")
    print("Treatment: unit_price")
    print("Outcome: quantity")
    print("Effect heterogeneity: product and time")
    print("Potential confounders: product, time, country")
    print("\nCausal graph:")
    print(CAUSAL_GRAPH)
