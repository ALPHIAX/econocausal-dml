# 📈 EconoCausal

### Dynamic Pricing & Causal Inference with Double Machine Learning

<p align="center">
  <strong>From Predictive Analytics to Causal Decision-Making</strong><br/>
  An open-source project exploring causal inference, personalized pricing, and data-driven promotional strategies.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/EconML-Causal%20ML-orange?style=for-the-badge" alt="EconML"/>
  <img src="https://img.shields.io/badge/DoWhy-Causal%20Inference-2E8B57?style=for-the-badge" alt="DoWhy"/>
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React"/>
  <img src="https://img.shields.io/badge/License-MIT-7952B3?style=for-the-badge" alt="MIT License"/>
</p>

---

## 📌 Overview

Traditional machine learning identifies patterns and predicts outcomes. However, prediction alone cannot answer an important business question:

**Did a discount actually cause a customer to purchase, or would the customer have purchased anyway?**

**EconoCausal** aims to bridge this gap by combining causal inference with machine learning. The project explores how Double Machine Learning (DML) and treatment-effect estimation can support better pricing and promotional decisions.

The long-term goal is to estimate the incremental impact of promotional interventions, identify customers who may respond to offers, and support budget-aware marketing strategies.

> **Core principle:** Predicting what a customer will do is different from estimating what will happen because of an intervention.

## 🎯 Business Problem

Businesses often distribute discounts broadly without knowing their incremental impact. This can lead to unnecessary margin loss, inefficient promotional spending, and missed opportunities.

EconoCausal explores three key questions:

* **Incrementality:** Does a promotional intervention change purchasing behavior?
* **Targeting:** Which customer groups are most likely to respond to an intervention?
* **Optimization:** How can promotional budgets be allocated to improve expected business outcomes?

## 💡 Proposed Solution

The project is designed around a causal analysis workflow:

1. Collect and preprocess observational datasets.
2. Explore data quality, customer behavior, prices, and transaction patterns.
3. Define a causal question, treatment, outcome, and adjustment variables.
4. Represent assumptions using a causal Directed Acyclic Graph (DAG).
5. Estimate treatment effects using suitable Double Machine Learning methods.
6. Evaluate model reliability and, where appropriate, heterogeneous treatment effects.
7. Explore budget-constrained promotional recommendations.

The validity of causal conclusions depends on the research design, data quality, overlap, and assumptions about confounding. DML does not automatically eliminate unobserved confounding.

## 🧠 Key Concepts

| Concept                         | Purpose                                                                                       |
| ------------------------------- | --------------------------------------------------------------------------------------------- |
| Causal inference                | Estimates the effect of an intervention rather than only predicting outcomes                  |
| Double Machine Learning         | Uses machine learning to estimate nuisance functions and construct treatment-effect estimates |
| Treatment effect                | Quantifies the difference in an outcome associated with a defined intervention                |
| Heterogeneous treatment effects | Examines how treatment effects vary across individuals or groups                              |
| Causal DAG                      | Documents assumptions about relationships among treatments, outcomes, and confounders         |
| Budget optimization             | Explores intervention allocation subject to business constraints                              |

**Important distinction:** Individual Treatment Effects (ITEs) are generally not directly observable for a single person. Individualized decisions require additional assumptions and careful validation.

**## 🗂️ Datasets**

The repository includes data preparation and exploration work for two different datasets.

**### 1. Online Retail II — Selected Primary Dataset**

A historical retail transaction dataset containing invoice identifiers, product codes, descriptions, quantities, invoice dates, prices, customer identifiers, and countries.

| Attribute                  | Details                                      |
| -------------------------- | -------------------------------------------- |
| Raw file                   | `Data/Raw/online_retail_II.xlsx`             |
| Source format              | Excel workbook                               |
| Worksheets                 | 2009–2010 and 2010–2011                      |
| Initial combined row count | 1,067,371                                    |
| Canonical processed output | `Data/Processed/cleaned_online_retail_dml.csv` |
| Final processed row count  | 793,609                                      |
| Main use                   | Product-level pricing and quantity analysis |

The preprocessing pipeline combines the available worksheets, standardizes column names, removes duplicate records, converts relevant data types, handles missing values, filters invalid transaction records, and prepares the data for causal analysis.

The canonical processed dataset contains 793,609 rows and 10 columns. It serves as the primary input for the Double Machine Learning (DML) pipeline.

### Causal Research Design

The current DML analysis uses the following variable definitions:

| Causal role | Dataset column / features | Description |
|---|---|---|
| Treatment (T) | `unit_price` | Continuous product price |
| Outcome (Y) | `quantity` | Transaction-level quantity purchased |
| Effect modifiers (X) | `stock_code` and temporal features | Product and time characteristics used to model treatment-effect heterogeneity |
| Adjustment variables (W) | `country` | Country-level features used for confounding adjustment |

The causal DAG represents product, time, and country as potential common causes of price and quantity. Product and time features are included in the DML effect-modifier matrix, while country is included in the adjustment matrix. The DoWhy audit uses product, country, and temporal features as common causes.

These are provisional observational causal assumptions. The analysis does not establish that all confounders have been measured or that price is free from endogeneity.

### Current DML Implementation

The current DML pipeline estimates heterogeneous treatment effects for product prices using the Online Retail II transaction dataset.

**Research question:**

How does a change in product price affect the quantity purchased per transaction, and how does this effect vary across products and time?

#### Variable definitions

| Role | Variable | Description |
|---|---|---|
| Treatment (T) | `unit_price` | Continuous product price |
| Outcome (Y) | `quantity` | Quantity purchased per transaction-product line |
| Effect modifiers (X) | Product and temporal features | Product indicators and temporal characteristics |
| Adjustment variables (W) | `country` | One-hot encoded country indicators |

#### Model configuration

| Component | Configuration |
|---|---|
| DML estimator | EconML `NonParamDML` |
| Final-stage estimator | `Ridge(alpha=1.0)` |
| Outcome nuisance model | LightGBM Regressor |
| Treatment nuisance model | LightGBM Regressor |
| Treatment type | Continuous |
| Cross-fitting | GroupKFold, 2 folds |
| External validation split | GroupShuffleSplit, 80/20 by invoice |
| Random state | 42 |

#### Feature construction

- X contains 107 features: 100 product indicators and 7 temporal features.
- W contains 41 country indicators.
- Product-specific temporal interaction features were removed in the final simplified specification.
- Invoice identifiers are used for grouped splitting, not as model features.
- `total_amount` is excluded because it is derived from price multiplied by quantity.

#### Training results

The completed training run produced:

| Metric | Result |
|---|---:|
| Selected products | 100 |
| Selected observations | 151,378 |
| Training observations | 120,892 |
| Test observations | 30,486 |
| Training invoices | 23,576 |
| Test invoices | 5,894 |
| Invoice overlap | 0 |
| Generated CATE estimates | 30,486 |

#### CATE output

The model saves test-set treatment-effect estimates to:

`Data/Processed/dml_ite_results.csv`

The output contains:

- `invoice_id`
- `stock_code`
- `country`
- `unit_price`
- `quantity`
- `estimated_cate`

The generated file contains 30,486 observations, with no missing or non-finite CATE estimates.

The observed mean estimated CATE was -56.67, and the median was -39.77. These estimates are exploratory and should not be interpreted as established causal effects without further identification and robustness assessment.

#### Interpretation and limitations

The current model estimates conditional marginal treatment effects for observed transaction-product lines. It does not estimate purchase probability or total market demand.

The analysis remains subject to potential unobserved confounding, endogenous pricing, limited within-product price variation, and model specification sensitivity. The prediction-sample outlier sensitivity check produced a similar mean CATE (-56.26), but this does not establish robustness to outliers during model training.

The final estimator differs from the earlier SparseLinearDML implementation. Results from the two specifications should not be treated as directly interchangeable.


**\*\*Data considerations:\*\***

* Missing customer identifiers and other essential fields require careful handling.
* Negative quantities may represent returns or reversals.
* Cancellation invoice prefixes require careful interpretation.
* Extreme quantities and prices require investigation rather than arbitrary removal.
* Transaction-level price and quantity associations do not, by themselves, establish a causal price effect.

The raw workbook is maintained separately from the generated processed datasets. The canonical processed CSV can be regenerated locally using the preprocessing pipeline.

**### 2. Starbucks Promotional Dataset — Explored Dataset**

The project also explored a Starbucks-style promotional dataset to investigate customer profiles, promotional offers, and customer interactions, consisting of:

| File              | Description                            |
| ----------------- | -------------------------------------- |
| `portfolio.json`  | Offer and promotion attributes         |
| `profile.json`    | Customer profile information           |
| `transcript.json` | Customer events and offer interactions |

The dataset provided an opportunity to explore promotional analytics, customer behavior, and offer interactions.

After evaluating both datasets against the project's primary research objective, Online Retail II was selected for the main DML-based dynamic pricing analysis because it provides transaction-level product prices, quantities, and timestamps.

These datasets are separate and have different schemas and analytical objectives. They are not merged into a single training dataset.

## 🧹 Data Preprocessing

The primary preprocessing script is:

`src/data_preprocessing.py`

Run it from the repository root:

```powershell
python src/data_preprocessing.py
```

The current preprocessing pipeline focuses on Online Retail II, which was selected as the primary dataset for the DML-based dynamic pricing analysis. The Starbucks promotional dataset was also explored during the initial research phase.
### Processing workflow

* Load the raw source datasets.
* Standardize column names and data types.
* Handle duplicate records and missing essential fields according to the current rules.
* Convert invoice dates into datetime values.
* Create transaction-value and data-quality indicator columns.
* Export the cleaned retail data to CSV.
* Preserve the raw source files for reproducibility.

### Output files

```text
Data/
├── Raw/
│   └── online_retail_II.xlsx
└── Processed/
    ├── online_retail_cleaned.csv
    ├── cleaned_online_retail_dml.csv
    └── dml_features.csv
```

The generated file `online_retail_cleaned.csv` is also written to `Data/Processed/`, but it is intentionally excluded from Git because it exceeds GitHub's standard file-size limit.


## 🏗️ Project Architecture

The intended architecture separates data preparation, causal analysis, and decision support.

```text
                  ┌─────────────────────────┐
                  │       Raw Data          │
                  │ │ Online Retail II      │    │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │    Data Preprocessing   │
                  │ Cleaning / Validation   │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Exploratory Analysis    │
                  │ Features / Data Quality │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Causal Research Design  │
                  │ Treatment / Outcome / X │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Causal Inference Engine │
                  │ DoWhy / EconML / DML    │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Effect Evaluation       │
                  │ Robustness / Diagnostics│
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────────┐
                  │ Decision Support            │
                  │ │ Planned Optimization / UI││
                  └─────────────────────────────┘
```

### Planned and supporting modules


| Module                                      | Responsibility                                      | Status      |
|---------------------------------------------|-----------------------------------------------------|-------------|
| `src/data_preprocessing.py`                 | Raw-data loading and preprocessing                  | Implemented |
| `src/validate_dml_data.py`                  | Processed dataset validation                        | Implemented |
| `src/dml_features.py`                       | Feature engineering for DML                         | Implemented |
| `src/causal_dag.py`                         | Causal graph definition and assumptions             | Implemented |
| `src/train_dml_model.py`                    | NonParamDML training and CATE estimation            | Implemented |
| `src/dowhy_causal_audit.py`                 | DoWhy causal audit and refutation tests             | Implemented |
| `src/product_price_support_diagnostics.py`  | Product-level price variation diagnostics           | Implemented |
| `src/customer_diagnostics.py`               | Customer-level price variation analysis             | Implemented |
| `src/nuisance_diagnostics.py`               | Treatment and outcome nuisance-model diagnostics    | Implemented |
| `src/overlap_diagnostics.py`                | Continuous-treatment overlap diagnostics            | Implemented |
| `src/product_time_diagnostics.py`           | Product and temporal variation analysis             | Implemented |
| `src/sample_selection_diagnostics.py`       | Eligible product cohort validation                  | Implemented |
| `src/stage2_outlier_assessment.py`          | Outlier assessment                                  | Implemented |
| `src/stage2_price_overlap.py`               | Product-level price-support assessment              | Implemented |
| `src/prepare_qini_data.py`                  | Preparation of CATE-ranking diagnostic data         | Implemented |
| `src/qini_analysis.py`                      | Exploratory CATE-ranking diagnostic                 | Implemented |
| Budget optimization                         | Budget-constrained decision optimization            | Planned     |
| Interactive dashboard                       | Visualization and decision-support interface        | Planned     |

The presence of a planned module in this architecture does not necessarily mean its implementation is complete.

## 🛠️ Technology Stack

* **Python:** Data preparation, causal analysis, and modeling
* **Pandas / NumPy:** Data manipulation and numerical computation
* **scikit-learn:** Machine learning estimators and preprocessing
* **LightGBM:** Nuisance-model estimation in the DML pipeline
* **DoWhy:** Causal modeling, identification, and refutation workflows
* **EconML:** Double Machine Learning and treatment-effect estimation
* **SciPy / CVXPY:** Numerical and constrained optimization (planned decision-support work)
* **Matplotlib / Plotly:** Diagnostics and visualization
* **React / Plotly.js:** Planned interactive dashboard

## 🚀 Getting Started

### Prerequisites

* * Python 3.11
* Git
* The raw datasets required by the selected preprocessing pipeline

### 1. Clone the repository

```bash
git clone https://github.com/ALPHIAX/econocausal-dml.git
cd econocausal-dml
```

### 2. Switch to the data-preparation branch

```bash
git checkout Anshika_DML

### 3. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by your system's execution policy, consult your local Python environment instructions rather than changing system policy unnecessarily.

### 4. Install dependencies

Install the project's declared dependencies:

```powershell
python -m pip install -r requirements.txt
```

### 5. Prepare the data

Ensure the expected raw files exist in `Data/Raw/`, then run:

```powershell
python src/data_preprocessing.py
```

The script should create the processed outputs under `Data/Processed/`.

To verify the retail output:

```powershell
python -c "import pandas as pd; d=pd.read_csv('Data/Processed/online_retail_cleaned.csv'); print('Shape:', d.shape); print(d.head())"
```
### 6. Run the DML pipeline

After generating the canonical processed dataset, execute:

```powershell
python src/train_dml_model.py
```

The script performs:

1. Eligible product selection.
2. Product and temporal feature construction.
3. Country adjustment-variable encoding.
4. Invoice-aware training and test splitting.
5. Grouped cross-fitting.
6. Double Machine Learning estimation.
7. CATE prediction and finite-value validation.
8. CSV export.

**Output:**

```text
Data/Processed/dml_ite_results.csv
```

### 7. Run diagnostic scripts

The following scripts provide supporting analysis:

```powershell
python src/sample_selection_diagnostics.py
python src/stage2_outlier_assessment.py
python src/stage2_price_overlap.py
python src/customer_diagnostics.py
python src/product_time_diagnostics.py
python src/overlap_diagnostics.py
python src/nuisance_diagnostics.py
```

Run them from the repository root with the project's virtual environment activated.


## 📊 Research and Evaluation Roadmap

### Phase 1 — Data Collection & Preprocessing

* [x] Add the Online Retail II raw workbook.
* [x] Implement the initial retail preprocessing pipeline.
* [x] Generate a locally processed transaction dataset.
* [x] Implement repeatable data-validation checks.
* [x] Document data-quality decisions and exclusions.

### Phase 2 — Exploratory Data Analysis

* [ ] Examine sales, quantities, prices, returns, and time patterns.
* [ ] Investigate missing customer IDs and unusual transactions.
* [ ] Identify a suitable unit of analysis.
* [ ] Document dataset limitations.

### Phase 3 — Causal Research Design

* [ ] Formulate a defensible causal question.
* [ ] Define treatment, outcome, and adjustment variables.
* [ ] Draw a causal DAG and document assumptions.
* [ ] Assess confounding, overlap, and data suitability.

### Phase 4 — DML Estimation

* [x] Implement an appropriate DML estimator.
* [x] Select and evaluate nuisance models.
* [x] Estimate heterogeneous treatment effects.
* [x] Generate and validate CATE predictions.
* [x] Conduct initial outlier and price-support diagnostics.
* [ ] Complete additional model robustness and identification assessment.

### Phase 5 — Decision Support

* [ ] Explore heterogeneous treatment effects where identifiable.
* [ ] Evaluate business costs and expected benefits.
* [ ] Design a constrained allocation strategy if supported by the estimates.
* [ ] Develop the interactive dashboard.

## ⚠️ Scientific and Business Limitations

EconoCausal is an analytical project, not a guarantee of causal identification or profitable pricing decisions.

* Observational data may contain unmeasured confounding.
* Price and quantity can influence one another through demand, promotions, and product characteristics.
* DML relies on assumptions and suitable nuisance-model estimation; it does not automatically remove every hidden confounder.
* Reliable personalized effects require sufficient data and additional identification assumptions.
* Historical retail transactions alone do not necessarily identify the causal impact of a hypothetical discount.
* Pricing recommendations should be evaluated for uncertainty, profitability, customer fairness, and practical constraints before deployment.

Where possible, future recommendations should be validated through controlled experiments or credible quasi-experimental designs.

## 🤝 Contributing

Contributions are welcome.

1. Create a feature branch.
2. Keep raw data unchanged.
3. Document preprocessing decisions and assumptions.
4. Add validation checks for data transformations.
5. Test changes before submitting a pull request.

Avoid committing large generated datasets or sensitive information. Follow the repository's `.gitignore` rules.

## 📄 License

This project declares an MIT license. See the repository's `LICENSE` file for the applicable terms.

## 👥 Project Goal

EconoCausal aims to connect machine learning, econometric reasoning, and decision optimization to support more informed pricing and promotional strategies.

**Better predictions are useful. Better causal decisions are the goal.**
