#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
forecast.py

This script uses the trained models from cooling.py and heating.py to make predictions
on new data. It applies the appropriate feature engineering for each model
and combines the predictions into a single output.

The script:
1. Imports the existing trained models from cooling.py and heating.py
2. Reads the dataset from forecast_daily.csv
3. Creates two derived datasets with the correct feature engineering
4. Applies the respective models to these prepared datasets
5. Combines the results into a single dataframe
6. Saves the combined result
"""

import pandas as pd
import numpy as np
import os
from os import path
import pickle
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler

# Import feature engineering functions from cooling.py and heating.py
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from predictions.cooling import preprocess_data as cooling_preprocess_data
from predictions.cooling import add_temporal_features as cooling_add_temporal_features
from predictions.heating import preprocess_data as heating_preprocess_data
from predictions.heating import add_temporal_features as heating_add_temporal_features

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console only
    ]
)
logger = logging.getLogger(__name__)

# Define base path
BASE_PATH = path.dirname(path.dirname(path.abspath(__file__)))

# Define paths
FORECAST_DATA_PATH = path.join(BASE_PATH, 'data', 'csv-daily', 'forecast_daily.csv')
COOLING_MODEL_PATH = path.join(BASE_PATH, 'predictions', 'best-model', 'cooling', 'best_model_cooling.pkl')
HEATING_MODEL_PATH = path.join(BASE_PATH, 'predictions', 'best-model', 'heating', 'best_model_heating.pkl')
OUTPUT_PATH = path.join(BASE_PATH, 'predictions', 'forecast_predictions.csv')

def load_data(data_path):
    """
    Load the forecast data from a CSV file

    Parameters:
    data_path (str): Path to the forecast data CSV file

    Returns:
    pd.DataFrame: Loaded data
    """
    try:
        logger.info(f"Loading forecast data from: {data_path}")
        # Check if the file exists
        if not os.path.exists(data_path):
            logger.error(f"File does not exist: {data_path}")
            return None
        data = pd.read_csv(data_path)
        logger.info(f"Forecast data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        return data
    except Exception as e:
        logger.error(f"Error loading forecast data: {e}")
        return None

def load_model(model_path):
    """
    Load a trained model from a pickle file

    Parameters:
    model_path (str): Path to the model pickle file

    Returns:
    object: Loaded model
    """
    try:
        logger.info(f"Loading model from: {model_path}")
        # Check if the file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file does not exist: {model_path}")
            return None
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model successfully loaded")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def preprocess_cooling_data(df):
    """
    Apply cooling-specific feature engineering to the forecast data

    Parameters:
    df (pd.DataFrame): Forecast data

    Returns:
    pd.DataFrame: Processed data ready for cooling model prediction
    """
    logger.info("Applying cooling-specific feature engineering")

    # Use imported functions from cooling.py
    # Step 1: Basic preprocessing - use a dummy target_col not present in the data
    processed_df = cooling_preprocess_data(df, target_col='dummy_target_not_in_data')
    # Step 2: Add temporal features
    processed_df = cooling_add_temporal_features(processed_df)

    # Drop date column for prediction
    features = processed_df.drop(columns=['date'], errors='ignore')

    logger.info(f"Cooling feature engineering completed: {features.shape[1]} features")
    return features

def preprocess_heating_data(df):
    """
    Apply heating-specific feature engineering to the forecast data

    Parameters:
    df (pd.DataFrame): Forecast data

    Returns:
    pd.DataFrame: Processed data ready for heating model prediction
    """
    logger.info("Applying heating-specific feature engineering")

    # Use imported functions from heating.py
    # Step 1: Basic preprocessing - use a dummy target_col not present in the data
    processed_df = heating_preprocess_data(df, target_col='dummy_target_not_in_data')
    # Step 2: Add temporal features
    processed_df = heating_add_temporal_features(processed_df)

    # Drop date column for prediction
    features = processed_df.drop(columns=['date'], errors='ignore')

    logger.info(f"Heating feature engineering completed: {features.shape[1]} features")
    return features

def ensure_model_features(model, features_df):
    """
    Ensure that the features match what the model expects

    Parameters:
    model: The trained model
    features_df (pd.DataFrame): The features for prediction

    Returns:
    pd.DataFrame: Features dataframe with the correct columns for the model
    """
    # Check if the model has feature_names_ (some models don't)
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        logger.info(f"Model expects {len(expected_features)} features")

        # Check for missing features
        missing_features = [f for f in expected_features if f not in features_df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features with value 0
            for feature in missing_features:
                features_df[feature] = 0

        # Check for extra features not expected by the model
        extra_features = [f for f in features_df.columns if f not in expected_features]
        if extra_features:
            logger.warning(f"Extra features being removed: {extra_features}")
            features_df = features_df.drop(columns=extra_features)

        # Ensure column order matches what the model expects
        features_df = features_df[expected_features]

    return features_df

def make_predictions():
    """
    Main function to load data and models, prepare data, make predictions, and save results
    """
    logger.info("Starting forecast prediction process")

    # Load forecast data
    forecast_data = load_data(FORECAST_DATA_PATH)
    if forecast_data is None:
        logger.error("Could not load forecast data. Stopping.")
        return

    # Keep a copy of the dates for the final output
    dates = forecast_data['date'].copy()

    # Voeg temperatuur trend features toe
    for window in [1, 3, 4]:
        forecast_data[f'temp_avg_{window}d'] = forecast_data['temperatuur_avg'].rolling(window=window, min_periods=1).mean()
        forecast_data[f'temp_trend_{window}'] = forecast_data[f'temp_avg_{window}d'] - forecast_data[f'temp_avg_{window}d'].shift(1)

    # Laad modellen
    cooling_model = load_model(COOLING_MODEL_PATH)
    heating_model = load_model(HEATING_MODEL_PATH)

    if cooling_model is None or heating_model is None:
        logger.error("Kon een of beide modellen niet laden. Stoppen.")
        return

    # Bereid data voor voor cooling model
    cooling_features = preprocess_cooling_data(forecast_data)

    # Schaal cooling features (StandardScaler zoals gebruikt in cooling.py)
    cooling_scaler = StandardScaler()
    cooling_features_scaled = pd.DataFrame(
        cooling_scaler.fit_transform(cooling_features),
        columns=cooling_features.columns
    )

    # Zorg ervoor dat de features overeenkomen met wat het cooling model verwacht
    cooling_features_scaled = ensure_model_features(cooling_model, cooling_features_scaled)

    # Bereid data voor voor heating model
    heating_features = preprocess_heating_data(forecast_data)

    # Schaal heating features (RobustScaler zoals gebruikt in heating.py)
    heating_scaler = RobustScaler()
    heating_features_scaled = pd.DataFrame(
        heating_scaler.fit_transform(heating_features),
        columns=heating_features.columns
    )

    # Zorg ervoor dat de features overeenkomen met wat het heating model verwacht
    heating_features_scaled = ensure_model_features(heating_model, heating_features_scaled)

    # Maak voorspellingen
    logger.info("Maken van cooling fault voorspellingen")
    cooling_predictions = cooling_model.predict(cooling_features_scaled)

    logger.info("Maken van heating fault voorspellingen")
    heating_predictions = heating_model.predict(heating_features_scaled)

    # Rond voorspellingen af naar gehele getallen
    cooling_predictions_rounded = np.round(cooling_predictions).astype(int)
    heating_predictions_rounded = np.round(heating_predictions).astype(int)

    # Maak output dataframe
    output_df = pd.DataFrame({
        'date': dates,
        'cooling_fault_prediction': cooling_predictions_rounded,
        'heating_fault_prediction': heating_predictions_rounded
    })

    # Sla voorspellingen op
    try:
        output_df.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"Voorspellingen opgeslagen in: {OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"Fout bij het opslaan van voorspellingen: {e}")

    logger.info("Forecast voorspellingsproces voltooid")

    return output_df

if __name__ == "__main__":
    logger.info("Start forecast.py")
    # Print huidige werkdirectory en basis pad voor debugging
    logger.info(f"Huidige werkdirectory: {os.getcwd()}")
    logger.info(f"Basis pad: {BASE_PATH}")
    logger.info(f"Forecast data pad: {FORECAST_DATA_PATH}")
    logger.info(f"Cooling model pad: {COOLING_MODEL_PATH}")
    logger.info(f"Heating model pad: {HEATING_MODEL_PATH}")

    predictions = make_predictions()
    if predictions is not None:
        logger.info(f"Voorspellingen gegenereerd voor {len(predictions)} dagen")
        # Toon eerste paar voorspellingen
        logger.info("\nVoorbeeld voorspellingen:")
        print(predictions.head())
    logger.info("forecast.py voltooid")
