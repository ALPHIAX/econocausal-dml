# DML Causal Design — Initial Specification

## 1. Project Objective

The EconoCausal project aims to study the relationship between pricing decisions and customer purchase behavior using causal inference and Double Machine Learning (DML).

The current DML workstream focuses on estimating the effect of product price on quantity purchased using historical retail transaction data.

---

## 2. Initial Causal Question

> What is the effect of product price on the quantity of products purchased, after accounting for relevant observed characteristics?

This is the initial causal question and will be validated and refined after exploratory data analysis and causal-structure review.

---

## 3. Initial Treatment and Outcome

### Treatment (T)

**Variable:** `price`

`price` is currently proposed as the treatment because the project focuses on understanding the effect of pricing on purchasing behavior.

The treatment is expected to be continuous rather than binary because product prices can take multiple numerical values.

### Outcome (Y)

**Variable:** `quantity`

`quantity` is currently proposed as the outcome because it represents the number of units purchased in a transaction line.

---

## 4. Unit of Analysis

One row represents one retail transaction line for a product on an invoice.

Therefore, the initial unit of analysis is:

**Transaction-product line**

This definition must remain consistent throughout preprocessing, causal analysis, model training, and interpretation.

---

## 5. Time Variable

**Variable:** `invoicedate`

The dataset contains historical transactions covering the 2009–2010 and 2010–2011 periods.

Time-related features may be required to account for temporal patterns such as changes in purchasing behavior or pricing over time.

---

## 6. Potential Covariates

The following variables are currently considered potential controls/covariates:

- Product
- Country
- Customer
- Time-related features

These variables are preliminary candidates only. Their final inclusion in the DML feature set must be determined after EDA and causal analysis.

Variables will not be included automatically simply because they are available in the dataset.

---

## 7. Causal Design Under Validation

The current working formulation is:

**Treatment (T):** `price`

**Outcome (Y):** `quantity`

**Covariates (X):** Product, country, customer, and appropriate time-related features

The proposed causal relationship is:

`price → quantity`

with relevant observed variables accounted for during causal estimation.

This formulation is preliminary and must be validated before final model training.

---

## 8. Validation Requirements Before DML

Before implementing the final DML model, the following must be investigated:

1. Whether `price` is an appropriate treatment variable.
2. Whether `quantity` is an appropriate causal outcome.
3. Whether the transaction-product line is the appropriate unit of analysis.
4. Which variables are genuine confounders or useful covariates.
5. Whether any variables introduce data leakage or post-treatment bias.
6. Whether sufficient variation exists in the treatment.
7. Whether the data provides adequate overlap/positivity.
8. How missing values, returns, and unusual transactions should be handled.
9. How time and temporal structure should be incorporated.
10. Whether the available observational data supports the intended causal interpretation.

---

## 9. DML Workstream

The planned DML workflow is:

1. Validate the causal question.
2. Define treatment, outcome, and covariates.
3. Construct and review the causal graph.
4. Prepare a leakage-safe modeling dataset.
5. Select appropriate nuisance models.
6. Apply cross-fitting.
7. Estimate the causal effect using DML.
8. Perform diagnostics and robustness checks.
9. Interpret the estimated effect within the limitations of the observational data.

---

## 10. Current Status

| Component | Status |
|---|---|
| Dataset selection | Finalized by team |
| EDA | Pending |
| Data dictionary | Pending |
| Treatment candidate | `price` |
| Outcome candidate | `quantity` |
| Causal DAG | Pending |
| Covariate selection | Pending validation |
| DML implementation | Pending |
| DML model training | Pending |
| Causal diagnostics | Pending |

> **Important:** The treatment, outcome, covariates, and causal interpretation documented here are the current working specification. They are not being treated as final until EDA and causal validation are completed.