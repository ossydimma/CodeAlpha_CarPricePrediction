"""
predict.py

Loads the trained CatBoost model and generates Selling_Price predictions for
new, unseen car listings. Mirrors the exact preprocessing steps applied in
03_data_cleaning.ipynb and 04_feature_engineering.ipynb, so a new listing is
transformed the same way the training data was.

Note: row-level cleaning steps from 03_data_cleaning.ipynb (dropping
duplicates, dropping the 500,000km outlier) were data quality fixes specific
to the training set and are NOT applied here — a new listing is never
"duplicate" or "an outlier" in isolation, so those steps don't carry over
to inference.
"""

import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# Reference year used for Car_Age, matching 04_feature_engineering.ipynb
# (the training dataset's maximum Year value)
REFERENCE_YEAR = 2018

# Exact column order the model was trained on (04_feature_engineering.ipynb)
FEATURE_ORDER = [
    'Present_Price', 'Driven_kms', 'Selling_type', 'Transmission',
    'Owner', 'Car_Age', 'Fuel_Type_CNG', 'Fuel_Type_Diesel', 'Fuel_Type_Petrol'
]

VALID_FUEL_TYPES = {'Petrol', 'Diesel', 'CNG'}
VALID_SELLING_TYPES = {'Dealer', 'Individual'}
VALID_TRANSMISSIONS = {'Manual', 'Automatic'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'catboost_model.cbm')

def load_model(model_path: str = MODEL_PATH) -> CatBoostRegressor:
    """Load the trained CatBoost model from disk."""
    model = CatBoostRegressor()
    model.load_model(model_path)
    return model

def _validate_input(row: dict) -> None:
    required = ['Year', 'Present_Price', 'Driven_kms', 'Fuel_Type',
                'Selling_type', 'Transmission', 'Owner']
    missing = [f for f in required if f not in row]
    if missing:
        raise ValueError(f"Missing required field(s): {missing}")

    if row['Fuel_Type'] not in VALID_FUEL_TYPES:
        raise ValueError(f"Fuel_Type must be one of {VALID_FUEL_TYPES}, got '{row['Fuel_Type']}'")
    if row['Selling_type'] not in VALID_SELLING_TYPES:
        raise ValueError(f"Selling_type must be one of {VALID_SELLING_TYPES}, got '{row['Selling_type']}'")
    if row['Transmission'] not in VALID_TRANSMISSIONS:
        raise ValueError(f"Transmission must be one of {VALID_TRANSMISSIONS}, got '{row['Transmission']}'")
    

def preprocess(data) -> pd.DataFrame:
    """
    Convert raw input (dict, list of dicts, or DataFrame) into the exact
    feature matrix the model expects.

    Expected raw fields per row:
        Year, Present_Price, Driven_kms, Fuel_Type ('Petrol'/'Diesel'/'CNG'),
        Selling_type ('Dealer'/'Individual'),
        Transmission ('Manual'/'Automatic'), Owner (0/1/3)
    """
    if isinstance(data, dict):
        data = [data]
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()

    for _, row in df.iterrows():
        _validate_input(row.to_dict())

    # Car_Age (04_feature_engineering.ipynb)
    df['Car_Age'] = REFERENCE_YEAR - df['Year']

    # Binary encoding (04_feature_engineering.ipynb)
    df['Selling_type'] = np.where(df['Selling_type'] == 'Dealer', 0, 1)
    df['Transmission'] = np.where(df['Transmission'] == 'Manual', 0, 1)

    # Fuel_Type one-hot — built manually rather than pd.get_dummies, since
    # get_dummies on a small/single-row batch would only create columns for
    # fuel types actually present, silently dropping the others the model
    # expects. Explicit construction guarantees all 3 columns always exist.
    df['Fuel_Type_CNG'] = (df['Fuel_Type'] == 'CNG').astype(int)
    df['Fuel_Type_Diesel'] = (df['Fuel_Type'] == 'Diesel').astype(int)
    df['Fuel_Type_Petrol'] = (df['Fuel_Type'] == 'Petrol').astype(int)

    # Reindex to the exact training column order — also protects against
    # any missing/extra columns causing a silent misalignment
    return df[FEATURE_ORDER]

def predict_price(data, model: CatBoostRegressor = None) -> np.ndarray:
    """
    Predict Selling_Price (in Lakhs) for one or more new car listings.

    data: dict (single listing), list of dicts, or DataFrame with raw fields
    model: optional pre-loaded model, to avoid reloading on every call
    """
    if model is None:
        model = load_model()

    X = preprocess(data)
    predictions = model.predict(X)
    return np.round(predictions, 2)

if __name__ == '__main__':
    # Example usage
    sample_listing = {
        'Year': 2015,
        'Present_Price': 9.85,
        'Driven_kms': 40000,
        'Fuel_Type': 'Petrol',
        'Selling_type': 'Dealer',
        'Transmission': 'Manual',
        'Owner': 0
    }

    model = load_model()
    predicted_price = predict_price(sample_listing, model=model)
    print(f"Predicted Selling_Price: {predicted_price[0]} Lakhs")