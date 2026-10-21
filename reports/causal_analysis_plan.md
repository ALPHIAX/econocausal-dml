# EconoCausal — Causal Analysis Plan

## 1. Business question

How does the observed unit price relate to the quantity purchased for a product, after accounting for observed product and time characteristics?

## 2. Dataset and unit of analysis

- Dataset: Online Retail II
- Unit: One transaction line
- Period: December 2009 to December 2011

## 3. Variables

- Treatment (T): `log_price`
- Outcome (Y): `log_quantity`
- Candidate controls (X): product identity, year, month, day of week, hour, and country
- Customer ID: optional, because some records lack this identifier

## 4. Estimand

Initially estimate an average treatment effect of a change in log price on log quantity, subject to the selected estimator and identification assumptions.

## 5. Identification limitations

The dataset does not directly record inventory availability or explicit promotion exposure. Price may also be influenced by order size, customer type, and business rules. These factors can bias the estimated effect.

## 6. Validation plan

- Inspect extreme prices and quantities.
- Check treatment overlap within products and time periods.
- Use cross-fitting that respects temporal structure where feasible.
- Compare reasonable model specifications.
- Document uncertainty and causal limitations.

## 7. Interpretation

Treat the estimate as observational evidence, not proof of causality. Do not use it for price optimization until the causal audit is complete.