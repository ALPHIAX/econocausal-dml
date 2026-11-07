"""
Provisional causal DAG for EconoCausal.

Research question:
How does product price affect quantity purchased,
and how does this effect vary across products and time?

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

    customer_characteristics -> price;
    customer_characteristics -> quantity;

    price -> quantity;
}
"""


def get_causal_graph():
    return CAUSAL_GRAPH


if __name__ == "__main__":
    print("EconoCausal: Provisional Causal DAG")
    print("-----------------------------------")
    print("Treatment: price")
    print("Outcome: quantity")
    print("Effect heterogeneity: product and time")
    print("Potential confounders: product, time, country, customer")
    print("\nCausal graph:")
    print(CAUSAL_GRAPH)