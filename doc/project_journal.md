## 2026-07-01

Started Car Price Prediction project — CodeAlpha internship Task 3.

### Dataset
301 rows × 9 columns. Single CSV file (car_data.csv).
Prices are in Indian Lakhs (1 Lakh = 100,000 rupees) — confirmed from value ranges.
Target variable: Selling_Price (what the seller is asking for the car).

### Column observations
Present_Price is the current ex-showroom price of the same model brand new.
It is always >= Selling_Price, which makes sense — used cars sell below new price.
This relationship between Present_Price and Selling_Price will likely be one of
the strongest signals in the model.

Driven_kms ranges from 500 to 500,000 — extreme right tail.
A car with 500,000km is an outlier that needs investigation in cleaning.

Owner is encoded as 0=first owner, 1=second, 3=fourth.
96.3% of listings are first-owner cars — very limited variance.
Value 2 (third owner) is completely absent from the dataset.
This column will likely have weak predictive power.

Fuel_Type has only 2 CNG entries (0.7%) — may need to be merged or dropped.
Transmission is 86.7% Manual — Automatic is the minority class.

### Duplicates
2 fully identical rows found across all 9 columns — data entry errors.
Decision: drop in cleaning notebook.

95 rows share the same Car_Name + Year combination — this is expected.
Multiple sellers listing the same model and year with different prices,
mileage and conditions. These are legitimate and will be kept.

### Year range
2003 to 2018 — 15 year span.
Age at listing ranges from brand new (0 years) to 15 years old.
A Car_Age feature will be engineered during feature engineering.

### Key flags for cleaning
- Drop 2 duplicate rows
- Investigate 500,000km outlier in Driven_kms
- Decide on CNG (2 entries) — merge into Other or drop
- Selling_Price is right-skewed (mean 4.66, max 35) — log transform likely needed

### Next step
EDA — distributions, price depreciation by age and mileage,
impact of fuel type, transmission, and seller type on price.