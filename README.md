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

## 🗂️ Datasets

The repository includes data preparation work for two different datasets.

### 1. Online Retail II

A historical retail transaction dataset containing invoice identifiers, product codes, descriptions, quantities, invoice dates, prices, customer identifiers, and countries.

| Attribute                  | Details                                    |
| -------------------------- | ------------------------------------------ |
| Raw file                   | `Data/Raw/online_retail_II.xlsx`           |
| Source format              | Excel workbook                             |
| Worksheets                 | 2009–2010 and 2010–2011                    |
| Initial combined row count | 1,067,371                                  |
| Preprocessed output        | `Data/Processed/online_retail_cleaned.csv` |
| Main use                   | Retail transaction and pricing analysis    |

The preprocessing pipeline combines the available worksheets, standardizes column names, removes exact duplicate rows, converts relevant data types, filters records missing essential transaction fields, and creates transaction-value and data-quality indicators.

The resulting dataset contains approximately 1.05 million rows. The exact output depends on the preprocessing rules and input data.

**Data considerations:**

* Missing customer identifiers limit customer-level linkage.
* Negative quantities may represent returns or reversals.
* Cancellation invoice prefixes require careful interpretation.
* Zero and negative prices should be investigated before analytical exclusions.
* Transaction-level price and quantity associations do not, by themselves, establish a causal price effect.

The raw workbook is tracked separately from the generated CSV. The processed CSV is excluded from ordinary Git tracking because of its size and can be regenerated locally.

### 2. Starbucks Promotional Dataset

The existing preprocessing pipeline also supports the Starbucks-style promotional dataset, consisting of:

| File              | Description                            |
| ----------------- | -------------------------------------- |
| `portfolio.json`  | Offer and promotion attributes         |
| `profile.json`    | Customer profile information           |
| `transcript.json` | Customer events and offer interactions |

The pipeline handles missing demographic placeholders, date conversion, nested data structures, and duplicate records.

These files are separate from Online Retail II. The two datasets have different schemas and should not be merged without a clearly justified analytical design.

## 🧹 Data Preprocessing

The primary preprocessing script is:

`src/data_preprocessing.py`

Run it from the repository root:

```powershell
python src/data_preprocessing.py
```

The current pipeline performs the existing Starbucks preprocessing and the added Online Retail II preprocessing.

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
│   ├── online_retail_II.xlsx
│   ├── portfolio.json
│   ├── profile.json
│   └── transcript.json
└── Processed/
    ├── cleaned_portfolio.csv
    ├── cleaned_profile.csv
    └── cleaned_transcript.csv
```

The generated file `online_retail_cleaned.csv` is also written to `Data/Processed/`, but it is intentionally excluded from Git because it exceeds GitHub's standard file-size limit.

## 🏗️ Project Architecture

The intended architecture separates data preparation, causal analysis, and decision support.

```text
                  ┌─────────────────────────┐
                  │       Raw Data          │
                  │ Retail / Promotions     │
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
                  ┌─────────────────────────┐
                  │ Decision Support        │
                  │ Optimization / Dashboard│
                  └─────────────────────────┘
```

### Planned and supporting modules

| Module                      | Responsibility                             |
| --------------------------- | ------------------------------------------ |
| `src/data_preprocessing.py` | Raw-data loading and preprocessing         |
| `src/causal_dag.py`         | Causal graph definition and assumptions    |
| `src/dml_engine.py`         | Double Machine Learning estimation         |
| `src/optimizer.py`          | Budget-constrained decision optimization   |
| `notebooks/`                | Exploratory analysis and model experiments |

The presence of a planned module in this architecture does not necessarily mean its implementation is complete.

## 🛠️ Technology Stack

* **Python:** Data preparation and analysis
* **Pandas / NumPy:** Data manipulation and numerical computation
* **scikit-learn:** Machine learning estimators and preprocessing
* **DoWhy:** Causal modeling and identification workflows
* **EconML:** Machine learning-based treatment-effect estimation
* **SciPy:** Numerical optimization
* **React / Plotly.js:** Planned interactive dashboard and visualization layer

## 🚀 Getting Started

### Prerequisites

* Python 3.9 or later, subject to compatibility of the installed dependencies
* Git
* The raw datasets required by the selected preprocessing pipeline

### 1. Clone the repository

```bash
git clone https://github.com/ALPHIAX/econocausal-dml.git
cd econocausal-dml
```

### 2. Switch to the data-preparation branch

```bash
git checkout data-prep
```

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

The Excel reader also requires `openpyxl`. If it is not included in `requirements.txt`, install it with:

```powershell
python -m pip install openpyxl
```

For reproducible collaboration, record all required packages in `requirements.txt`.

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

## 📊 Research and Evaluation Roadmap

### Phase 1 — Data Collection & Preprocessing

* [x] Add the Online Retail II raw workbook.
* [x] Implement the initial retail preprocessing pipeline.
* [x] Generate a locally processed transaction dataset.
* [x] Commit and push the preprocessing code to the `data-prep` branch.
* [ ] Document data-quality decisions and exclusions.
* [ ] Add repeatable validation checks.

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

* [ ] Implement an appropriate DML estimator.
* [ ] Select and evaluate nuisance models.
* [ ] Estimate treatment effects with uncertainty intervals.
* [ ] Conduct sensitivity and robustness checks.

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
