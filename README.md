# 📈 EconoCausal: Dynamic Pricing via Double Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![EconML](https://img.shields.io/badge/EconML-Causal%20AI-orange?style=for-the-badge)
![DoWhy](https://img.shields.io/badge/DoWhy-Causal%20Graphs-green?style=for-the-badge)
![React](https://img.shields.io/badge/React-18.0-blue?style=for-the-badge&logo=react)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

---

## 🎯 Executive Overview

Standard predictive machine learning focuses on **correlation, not causation**. A predictive churn model might recommend offering a \$20 discount to a high-risk user, but fails to answer the critical strategic question: *"Would this user have purchased anyway, wasting the budget?"*

**EconoCausal** moves beyond traditional predictive AI into **Causal and Prescriptive AI**. By implementing **Double Machine Learning (DML)** and calculating the **Individual Treatment Effect (ITE)**, EconoCausal isolates exact causality from observational data. This enables marketing and enterprise finance teams to target **"Persuadables"** (users who convert *only* when given a discount), maximizing promotional campaign ROI.

---

## 💡 Core Use Case & Value Proposition

- **The Problem:** Blanket promotional discounts waste budget on customers who would convert organically (Lost Margins) or fail to sway non-persuadable users.
- **The Solution:** EconoCausal calculates the individual uplift for each user across candidate discount tiers (e.g., \$10 vs \$20) and uses a constrained optimization solver to allocate budgets with maximum efficiency.

---

## 🛠️ Key Architectural Modules

| Module | Technologies | Description |
| :--- | :--- | :--- |
| **Causal Inference Engine** | `DoWhy`, `EconML` | Defines Directed Acyclic Graphs (DAGs) and runs Double/Debiased Machine Learning to strip out hidden confounders. |
| **Propensity Score Matching** | `scikit-learn` | Balances historical observational data to simulate randomized control trials (A/B tests). |
| **Prescriptive Solver** | `SciPy Optimization` | Solves budget-constrained linear/non-linear optimization problems to assign individual personalized discounts. |
| **Experimentation Dashboard** | `React`, `Plotly.js` | Interactive UI displaying Causal Uplift (Qini) curves, dynamic customer filters, and prescription matrices. |

---

## 🗓️ Development Roadmap

-## 🧹 Data Preprocessing & Pipeline

The data preparation phase ingests, cleans, and structures raw observational campaign logs into standardized tabular files for causal modeling:

* **`src/data_preprocessing.py`**: Automated pipeline that loads raw JSON files, handles missing demographic placeholders (e.g., age `118`), formats dates, and safely encodes complex nested lists/dictionaries to enable seamless deduplication.
* **Processed Datasets (`Data/Processed/`)**:
  * `cleaned_profile.csv`: Cleaned user demographics.
  * `cleaned_portfolio.csv`: Standardized offer and promotion specifications.
  * `cleaned_transcript.csv`: Cleaned customer interaction logs and event histories.

To execute the data cleaning pipeline locally:
```powershell
python src/data_preprocessing.py

## 📂 Repository Structure

```text
econocausal-dml/
├── Data/
│   ├── Raw/            # Initial observational campaign datasets
│   └── Processed/      # Cleaned and feature-engineered datasets
├── notebooks/          # Exploratory analysis and causal model prototyping
├── src/
│   ├── causal_dag.py   # DoWhy DAG graph initialization
│   ├── dml_engine.py   # EconML Double Machine Learning pipelines
│   └── optimizer.py    # SciPy budget allocation optimization solver
├── .gitignore
├── README.md
└── requirements.txt