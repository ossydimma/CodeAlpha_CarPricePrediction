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

## 2026-07-02

Completed EDA for Car Price Prediction (02_eda.ipynb).

### Distributions
Selling_Price and Present_Price are both right-skewed, confirming the log transform
flagged during data understanding. log(Selling_Price) is visibly closer to symmetric
(mean 0.91, range -2.30 to 3.56).

Driven_kms confirmed the 500,000km outlier stands out clearly and in isolation from
the rest of the distribution — will need a decision in cleaning (cap, remove, or keep).

### Price relationships
Present_Price vs Selling_Price correlation is 0.879 — by far the strongest predictor
in the dataset. Car_Age shows a clear negative relationship with Selling_Price, as
expected. Driven_kms shows a negative relationship too but noisier.

### Categorical effects
Diesel cars have double the median selling price of Petrol (7.75 vs 2.65).
Automatic transmission commands a premium over Manual (5.80 vs 3.25).
Dealer-sold cars have a far higher median price than Individual-sold (5.25 vs 0.515) —
this is the largest categorical split found and worth exploring in feature engineering.

### Multicollinearity noted
Year and Car_Age are perfectly inversely correlated by construction (Car_Age = 2018 -
Year). Only one will be kept for modelling — Car_Age, since it's more interpretable.

### Next step
Data cleaning — drop 2 duplicate rows, investigate the 500,000km Driven_kms outlier
and the negative-depreciation row, decide on CNG (2 entries: merge or drop).

## 2026-07-02

Completed data cleaning for Car Price Prediction (03_data_cleaning.ipynb).

### Duplicates
2 fully identical rows found (ertiga 2016, fortuner 2015) — confirmed with keep=False
before dropping. Dataset reduced from 301 to 299 rows.

### Driven_kms outlier
One 2008 Activa 3G listed with 500,000 km. Comparing it against a second Activa 3G
listing (500 km) in the dataset made the case clearer — 500,000 km implies ~50,000
km/year on a private two-wheeler, well above realistic use, and looks like a data
entry error (extra zeros) rather than genuine mileage. Row dropped. Dataset reduced
from 299 to 298 rows.

### Rare fuel type (CNG)
Only 2 CNG vehicles vs 238 Petrol and 58 Diesel. Retained — the observations are
valid, not a data quality issue. Small sample size may limit the model's ability to
learn CNG-specific patterns; deferred to feature engineering/modelling.

### Final dataset
298 rows x 9 columns, 0 missing values. Saved to ../data/processed/car_clean.csv.

### Next step
Feature engineering — create Car_Age, resolve Year/Car_Age collinearity, and encode
Fuel_Type, Selling_type, and Transmission.