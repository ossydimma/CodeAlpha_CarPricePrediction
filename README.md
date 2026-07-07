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

- Selling_Price is right-skewed (mean 4.66 Lakhs, max 35.00) — log transform confirmed necessary for modelling
- **Present_Price is the strongest predictor** — correlation of 0.879 with Selling_Price
- Car_Age shows a clear negative relationship with price; Driven_kms shows a noisier negative relationship
- Diesel cars sell for a median of 7.75 Lakhs vs 2.65 for Petrol
- Automatic transmission commands a premium over Manual (5.80 vs 3.25 median)
- Dealer-sold cars have a far higher median price than Individual sellers (5.25 vs 0.515) — the largest categorical split in the dataset
- Depreciation ranges from 1.07% to 89.46% — the lowest belongs to a near-new car, confirming Present_Price >= Selling_Price holds across the full dataset
- Year and Car_Age are perfectly collinear by construction — only Car_Age will be kept for modelling

---

## Data Cleaning Summary

- 2 fully identical duplicate rows dropped (301 → 299 rows)
- One 2008 Activa 3G with 500,000 km dropped as a likely data entry error — a second
  Activa 3G listing in the dataset shows only 500 km, and 500,000 km implies ~50,000
  km/year, unrealistic for a private two-wheeler (299 → 298 rows)
- Rare fuel type (CNG, 2 entries) retained — valid data, not a quality issue; small
  sample size deferred to feature engineering/modelling
- No missing values, no dtype corrections needed
- Final dataset: 298 rows x 9 columns, saved to `data/processed/car_clean.csv`

---

## Feature Engineering Summary

- Car_Age created (2018 - Year); Year dropped to resolve perfect collinearity
- Selling_type and Transmission binary-encoded (0/1)
- Fuel_Type one-hot encoded into 3 columns (CNG, Diesel, Petrol)
- Car_Name dropped (98 unique values, too sparse for 298 rows); Brand and
  Brand_Freq were engineered and tested but ultimately dropped — the grouping
  conflated cars and two-wheelers instead of capturing manufacturer, and
  correlation with Selling_Price was weak (-0.1158)
- Owner left unchanged (already numeric and ordinal)
- Final dataset: 298 rows x 10 columns, all numeric, saved to `data/processed/car_features.csv`

---

## Results

Final model: **CatBoost** (tuned via Optuna, 50 trials, 5-fold CV)

| Model | RMSE (Lakhs) | R² |
|-------|--------------|-----|
| CatBoost (tuned) | 1.1961 | 0.9346 |
| CatBoost (default) | 1.4259 | 0.9039 |
| Linear Regression | 1.8399 | 0.8579 |
| Mean Baseline | 4.9402 | -0.0132 |

Tuning reduced RMSE by 16.1% over default CatBoost parameters. Present_Price is by
far the strongest predictor (78.2% feature importance), followed by Car_Age (11.7%).

Best parameters: `depth=3, iterations=527, learning_rate=0.1343, l2_leaf_reg=6.74,
min_data_in_leaf=11`

---

## Model Evaluation

Out-of-fold RMSE: **1.3556 Lakhs** | R²: **0.9258** | MAE: **0.6364 Lakhs**

| Price Band | MAE | RMSE | Count |
|------------|-----|------|-------|
| Low | 0.172 | 0.256 | 76 |
| Mid-Low | 0.463 | 0.914 | 73 |
| Mid-High | 0.671 | 1.145 | 77 |
| High | 1.266 | 2.299 | 72 |

The model performs excellently on mainstream/budget cars and is meaningfully less
precise on premium vehicles (RMSE grows ~9x from the cheapest to most expensive
quartile) — a natural consequence of limited high-value examples in a 298-row
dataset. SHAP confirms Present_Price and Car_Age as the two dominant drivers of
every prediction. **Recommended use:** trust predictions directly under ~10 Lakhs;
treat predictions above that as directional and flag for human review.

---

## Workflow

| Phase | Notebook | Status |
|-------|----------|--------|
| Data understanding | 01_data_understanding.ipynb | ✅ Complete |
| Exploratory analysis | 02_eda.ipynb | ✅ Complete |
| Data cleaning | 03_data_cleaning.ipynb | ✅ Complete |
| Feature engineering | 04_feature_engineering.ipynb | ✅ Complete |
| Modelling | 05_modelling.ipynb | ✅ Complete |
| Evaluation | 06_evaluation.ipynb | ✅ Complete |

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