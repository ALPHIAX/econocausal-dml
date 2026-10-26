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

## 2. Invoicedate Analysis

### Variable
`invoicedate`

### Role
Temporal covariate / X candidate.

### Dataset Evidence

The transaction data was aggregated by month to examine whether the treatment and outcome distributions vary over time.

Observed monthly average prices varied across the study period. For example:

- January 2010: average `unit_price` ≈ 3.92
- November 2010: ≈ 3.04
- May 2011: ≈ 3.62
- July 2011: ≈ 2.84
- December 2011: ≈ 2.74

Monthly quantity distributions also varied. For example:

- January 2010: average `quantity` ≈ 17.25
- November 2010: ≈ 11.09
- January 2011: ≈ 16.60
- November 2011: ≈ 10.54
- December 2011: ≈ 16.84

### Decision

`invoicedate` is **retained as a temporal X candidate**.

### Reasoning

Both the treatment (`unit_price`) and outcome (`quantity`) distributions vary over time. Temporal information may therefore capture systematic changes in pricing and purchasing behavior that should be accounted for in the DML design.

### Modeling Consideration

The raw datetime value should not be treated as a simple categorical or continuous identifier. During preprocessing/modeling, meaningful temporal features may be derived, such as:

- year
- month
- day of week
- hour
- time trend

### Limitation

The observed temporal patterns establish association, not causation. Therefore, `invoicedate` is treated as an X candidate rather than being labeled a confirmed causal confounder.

## 3. Customer ID Analysis

### Variable
`customer_id`

### Role
Customer-level covariate / X candidate.

### Dataset Evidence

Customer-level variation was examined to determine whether customer identity is associated with both observed prices and purchased quantities.

Customers showed substantial variation in observed prices. For example:

- Customer `17841`: average `unit_price` ≈ 2.73, 81 unique prices
- Customer `14911`: average `unit_price` ≈ 5.14, 134 unique prices
- Customer `12748`: average `unit_price` ≈ 3.01, 120 unique prices
- Customer `14096`: average `unit_price` ≈ 6.52, 64 unique prices

Customer-level quantity behavior also varied substantially. For example:

- Customer `17841`: average `quantity` ≈ 2.95
- Customer `14911`: average `quantity` ≈ 13.34
- Customer `14646`: average `quantity` ≈ 95.40
- Customer `14156`: average `quantity` ≈ 41.05

### Decision

`customer_id` is **retained as an X candidate**, subject to appropriate representation during modeling.

### Reasoning

Observed customer-level differences exist in both pricing and purchasing behavior. Customer identity may therefore capture systematic differences that are relevant to the treatment-outcome relationship.

### Modeling Consideration

`customer_id` is an identifier and should **not be treated as a continuous numerical variable**.

The dataset contains many distinct customers, so the representation or encoding strategy must be selected carefully during the modeling stage.

### Limitation

The observed association between customer identity, price, and quantity does not by itself establish that customer identity is a causal confounder.

## 4. Country Analysis

### Variable
`country`

### Role
Geographical covariate / X candidate.

### Dataset Evidence

Transaction-level analysis shows substantial variation in observed prices across countries.

Examples of average `unit_price`:

- UK: ≈ 3.07
- Germany: ≈ 3.60
- France: ≈ 4.28
- EIRE: ≈ 5.32
- Portugal: ≈ 5.07
- Norway: ≈ 14.21

Quantity behavior also varies across countries.

Examples of average `quantity`:

- UK: ≈ 12.15
- Germany: ≈ 13.66
- France: ≈ 19.85
- EIRE: ≈ 20.44
- Netherlands: ≈ 75.48
- Australia: ≈ 57.47
- Sweden: ≈ 67.19
- Denmark: ≈ 298.15

### Decision

`country` is **retained as an X candidate**.

### Reasoning

The observed treatment (`unit_price`) and outcome (`quantity`) distributions differ across countries. Country may therefore capture geographical differences associated with both pricing and purchasing behavior.

### Modeling Consideration

`country` is categorical and should be represented using an appropriate categorical encoding rather than as a continuous numerical variable.

### Limitation

These cross-country differences establish observational association only. They do not prove that country causally determines either price or quantity.


## 5. Final Feature Decisions

| Column | DML Role | Decision |
|---|---|---|
| `unit_price` | Treatment (T) | Use as treatment |
| `quantity` | Outcome (Y) | Use as outcome |
| `stock_code` | Product-level covariate | Retain as X candidate |
| `invoicedate` | Temporal covariate | Retain as X candidate |
| `country` | Geographical covariate | Retain as X candidate |
| `customer_id` | Customer-level covariate | Retain as X candidate |
| `description` | Text/product description | Exclude from initial model |
| `invoice_id` | Transaction identifier | Exclude |
| `source_sheet` | Dataset metadata | Exclude |
| `total_amount` | Derived variable | Exclude |

### Excluded Variables

#### `description`

`description` is excluded from the initial DML feature set because it is high-dimensional text and overlaps substantially with product identity represented by `stock_code`.

The initial model will therefore use `stock_code` as the primary product-level representation. Text-based representations may be investigated separately if additional information is required.

#### `invoice_id`

`invoice_id` is a transaction identifier rather than a substantive explanatory variable. It is therefore excluded from the X feature set.

#### `source_sheet`

`source_sheet` identifies the source worksheet from which a transaction originated. It is dataset metadata rather than a substantive causal covariate and is excluded from the initial model.

#### `total_amount`

`total_amount` is calculated as:

`quantity × unit_price`

Because it is directly constructed from the treatment (`unit_price`) and outcome (`quantity`), including it as an X variable would introduce target/treatment leakage and is therefore inappropriate for the DML feature set.

### Overall Feature-Selection Principle

The selected X candidates are based on observed variation and their potential relevance to the treatment-outcome relationship. These decisions do not establish that every retained variable is a confirmed causal confounder.

Final inclusion and representation of the X variables will be determined during the DML modeling stage based on causal assumptions, data structure, encoding requirements, and model performance.

## Causal Audit: Assumptions and Limitations

### Causal Assumptions

The DoWhy audit identifies a backdoor adjustment estimand for the effect of `unit_price` on `quantity`, conditional on the selected covariates.

The analysis relies on the following assumptions:

1. **Conditional unconfoundedness**
   After conditioning on the selected covariates, there are no unobserved variables that simultaneously affect `unit_price` and `quantity`.

2. **Correct causal specification**
   The selected adjustment variables and their assumed causal relationships are appropriate for the pricing and purchasing process.

3. **Positivity / overlap**
   Sufficient variation in `unit_price` exists across the relevant covariate patterns to estimate the treatment effect.

4. **Valid outcome and treatment measurements**
   `unit_price` and `quantity` are assumed to be measured consistently and represent the intended treatment and outcome.

### Refutation Results

The DoWhy causal audit included three robustness checks:

- **Placebo treatment refutation:** the estimated effect changed from approximately `-0.9706` to `-0.0046`, indicating that the effect disappeared when the treatment was randomly permuted.
- **Random common cause refutation:** the estimate remained approximately `-0.9706` after adding a random common cause.
- **80% data subset refutation:** the estimate changed from approximately `-0.9706` to `-0.9666`, a change of approximately `0.4%`.

These tests provide robustness evidence for the current causal specification but do not prove that the estimate is free from hidden confounding.

### Limitations

- The analysis uses observational retail data rather than randomized experiments.
- Unobserved confounding cannot be ruled out from these tests alone.
- `stock_code` and `country` are currently represented using integer category codes in the DML implementation; these codes are representations rather than inherently ordered numerical variables.
- The estimated causal effect depends on the selected covariates, preprocessing decisions, and modeling assumptions.
- The DoWhy audit estimate is a separate audit estimate and does not replace the primary EconML DML estimate.

additional