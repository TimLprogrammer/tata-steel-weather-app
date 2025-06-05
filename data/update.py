import pandas as pd
from datetime import datetime, timedelta
import os
import logging
import sys
import subprocess
import shutil

# Import the weather data functions
from script.historical import get_historical_data
from script.forecast import combine_and_save_data

# Configure base paths
BASE_PATH = os.path.join(".", "main_project")
DATA_PATH = os.path.join(BASE_PATH, "data")

# Configure logging with simpler format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # Standardize date format
)

def clean_zero_columns(df, weather_params):
    """Remove columns where 50% or more of values are either 0/0.0 or empty (NaN)."""
    zero_cols = []
    nan_cols = []
    threshold = 0.5  # 50% threshold

    for col in df.columns:
        # Check for zeros (including '0', '0.0', '0.00')
        zero_count = df[col].astype(str).isin(['0', '0.0', '0.00']).mean()
        # Check for NaN values
        nan_count = df[col].isna().mean()

        if zero_count >= threshold:
            zero_cols.append(col)
        elif nan_count >= threshold:
            nan_cols.append(col)

    # Remove columns with too many zeros
    if zero_cols:
        logging.info(f"Removing columns with ≥50% zeros: {zero_cols}")
        logging.info("Zero percentages for removed columns:")
        for col in zero_cols:
            zero_percent = df[col].astype(str).isin(['0', '0.0', '0.00']).mean() * 100
            logging.info(f"{col}: {zero_percent:.1f}%")

    # Remove columns with too many NaN values
    if nan_cols:
        logging.info(f"Removing columns with ≥50% empty values: {nan_cols}")
        logging.info("Empty percentages for removed columns:")
        for col in nan_cols:
            nan_percent = df[col].isna().mean() * 100
            logging.info(f"{col}: {nan_percent:.1f}%")

    # Combine columns to remove
    cols_to_remove = zero_cols + nan_cols
    if cols_to_remove:
        df = df.drop(columns=cols_to_remove)
        # Update weather_params list
        weather_params = [param for param in weather_params if param not in cols_to_remove]
        logging.info(f"Removed total of {len(cols_to_remove)} columns")

    return df, weather_params

def standardize_dataframe(df):
    """Standardize DataFrame format"""
    # Convert date column to datetime if it isn't already
    df['date'] = pd.to_datetime(df['date'])

    # Ensure datetime is timezone-naive
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_localize(None)

    # Convert to string in standard format
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Ensure all numeric columns have consistent float format with exactly 6 decimal places
    for col in df.columns:
        if col != 'date':
            df[col] = df[col].astype(float).round(6)

    return df

def update_weather_data():
    """Update both historical and forecast weather data files"""
    weather_params = [
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
        'apparent_temperature', 'pressure_msl', 'surface_pressure',
        'cloud_cover', 'wind_speed_10m', 'wind_direction_10m',
        'wind_gusts_10m'
    ]

    data_dir = os.path.join(DATA_PATH, 'csv-api')
    os.makedirs(data_dir, exist_ok=True)

    historical_file = os.path.join(data_dir, 'historical.csv')
    forecast_file = os.path.join(data_dir, 'forecast.csv')

    try:
        logging.info("Fetching complete historical dataset...")
        historical_data = get_historical_data(start_date="2018-01-01")  # Changed start date
        historical_data = historical_data[['date'] + weather_params]
        historical_data = historical_data.dropna(subset=weather_params)
        historical_data.to_csv(historical_file, index=False)
        logging.info(f"Historical data saved to {historical_file}")

        # Use the new combined function that includes both historical (1 month) and forecast data
        logging.info("Fetching combined historical (1 month) and forecast data...")
        combine_and_save_data()  # This will save to forecast.csv with both historical and forecast data
        logging.info(f"Combined historical and forecast data saved to {forecast_file}")

    except Exception as e:
        logging.error(f"Error updating weather data: {str(e)}")
        raise

def run_script(script_path):
    """Run a Python script and return True if successful"""
    try:
        # Create a new environment with current environment variables
        env = os.environ.copy()

        # Run the script with the environment variables
        result = subprocess.run(
            [sys.executable, script_path],
            env=env,
            check=True,
            stdout=True,  # Changed to True to show output directly
            stderr=True   # Changed to True to show errors directly
        )

        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {script_path}: {str(e)}")
        if e.stderr:
            print(e.stderr.decode())
        return False

def main():
    # Use fixed path for uploaded Excel file
    notifications_path = os.path.join(DATA_PATH, "uploaded_data.xlsx")

    if not os.path.exists(notifications_path):
        logging.error(f"Uploaded Excel file not found: {notifications_path}")
        return

    # Create only necessary directory
    os.makedirs(os.path.join(DATA_PATH, 'notifications'), exist_ok=True)

    # Update weather data first
    logging.info("Updating weather data...")
    update_weather_data()

    # Set environment variables for paths
    os.environ['NOTIFICATIONS_PATH'] = notifications_path
    os.environ['DATA_ROOT'] = DATA_PATH

    # Define the sequence of scripts to run
    scripts_to_run = [
        os.path.join(DATA_PATH, 'script', 'classifie_vks.py'),
        os.path.join(DATA_PATH, 'script', 'process_daily_weather.py'),
        os.path.join(DATA_PATH, 'script', 'result.py')
    ]

    # Run each script in sequence
    for script in scripts_to_run:
        logging.info(f"\nRunning {os.path.basename(script)}...")
        if not os.path.exists(script):
            logging.error(f"Script not found: {script}")
            continue

        if not run_script(script):
            logging.error(f"Failed to run {script}. Stopping execution.")
            break

    logging.info("\nAll processing completed!")

if __name__ == "__main__":
    logging.info("Starting data update and processing pipeline...")
    main()
