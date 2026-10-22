# DML Feature Analysis

## 1. Stock Code Analysis

### Variable
`stock_code`

### Role
Product-level covariate / X candidate.

### Dataset Evidence

The DML dataset contains **4,631 unique stock codes**.

Product observation frequency is uneven:

| Minimum observations | Number of products |
|---|---:|
| ≥ 5 | 4,241 |
| ≥ 10 | 3,924 |
| ≥ 20 | 3,434 |
| ≥ 50 | 2,600 |
| ≥ 100 | 1,881 |

Price variation was also examined within each product:

- Products with more than one observed `unit_price`: **3,569 (77.07%)**
- Products with exactly one observed `unit_price`: **1,062 (22.93%)**

The correlation between the number of unique prices and the number of unique quantities within a product was **0.285233**.

### Decision

`stock_code` is **retained as an X candidate**.

### Reasoning

Products show substantial variation in observed prices and quantity behavior. Product identity may therefore contain information relevant to both the treatment (`unit_price`) and outcome (`quantity`) relationship.

### Modeling Consideration

`stock_code` should **not be treated as a continuous numerical variable** because it is a categorical product identifier.

There are 4,631 unique categories with uneven observation frequencies. The appropriate encoding or representation will be determined during the modeling stage.

### Limitation

These analyses establish observational associations and support the feature-selection decision. They do **not**, by themselves, prove that `stock_code` is a causal confounder.
