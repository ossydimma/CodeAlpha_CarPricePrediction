# Car Price Prediction

## Business Problem

Buyers and sellers in the used car market struggle to determine fair pricing.
Sellers risk underpricing their vehicle or failing to attract buyers by overpricing.
Buyers have no reliable way to know if an asking price is reasonable.

This project builds a machine learning model to predict the selling price of a
used car based on its characteristics — age, mileage, fuel type, transmission,
and current market value of the same model new.

---

## Dataset

**Source:** CodeAlpha Internship — Task 3

| File | Rows | Description |
|------|------|-------------|
| car_data.csv | 301 | Indian used car listings with price and features |

> Raw data files are not tracked in this repository.
> Download from the source and place in `data/raw/`.

**Target variable:** `Selling_Price` — asking price in Indian Lakhs

---

## Project Structure

```
Car_Price_Prediction/
│
├── data/
│   ├── raw/                        # Original CSV (not tracked by git)
│   └── processed/                  # Cleaned outputs (not tracked by git)
│
├── docs/
│   └── project_journal.md
│
├── images/                         # Charts saved from notebooks
├── models/                         # Saved model file
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modelling.ipynb
│   └── 06_evaluation.ipynb
│
├── src/
│   └── predict.py
│
├── requirements.txt
├── .gitignore
└── README.md

```

---

## Key Findings from EDA

*To be updated after EDA notebook is complete.*

---

## Data Cleaning Summary

*To be updated after cleaning notebook is complete.*

---

## Feature Engineering Summary

*To be updated after feature engineering is complete.*

---

## Results

*To be updated after modelling is complete.*

---

## Workflow

| Phase | Notebook | Status |
|-------|----------|--------|
| Data understanding | 01_data_understanding.ipynb | ✅ Complete |
| Exploratory analysis | 02_eda.ipynb | ⏳ Pending |
| Data cleaning | 03_data_cleaning.ipynb | ⏳ Pending |
| Feature engineering | 04_feature_engineering.ipynb | ⏳ Pending |
| Modelling | 05_modelling.ipynb | ⏳ Pending |
| Evaluation | 06_evaluation.ipynb | ⏳ Pending |

---

## Evaluation Metric

**RMSE and R²** — RMSE gives the average prediction error in Lakhs.
R² measures how much of the price variance the model explains.
Both reported at default scale and on log-transformed target.

---

## Tools and Libraries

Python · pandas · numpy · matplotlib · seaborn · scikit-learn · LightGBM · joblib

---

## Author

Built as part of the CodeAlpha Data Science Internship (June–July 2026).

[LinkedIn](https://www.linkedin.com/in/osita-jerry) · 
[GitHub](https://github.com/ossydimma/CodeAlpha_CarPricePrediction)