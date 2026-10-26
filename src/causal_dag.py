"""
Initial causal DAG for the EconoCausal project.

Causal question:
What is the effect of product price on quantity purchased?

The DAG represents the current causal assumptions.
These assumptions are provisional and must be validated
using the dataset, EDA, and business context.
"""


CAUSAL_GRAPH = """
digraph {
    product -> price;
    product -> quantity;

    time -> price;
    time -> quantity;

    price -> quantity;
}
"""


def get_causal_graph():
    """Return the current causal graph definition."""
    return CAUSAL_GRAPH


if __name__ == "__main__":
    print("Initial EconoCausal DAG")
    print("----------------------")
    print("Treatment: price")
    print("Outcome: quantity")
    print("Potential confounders: product, time")
    print("\nCausal graph:")
    print(CAUSAL_GRAPH)