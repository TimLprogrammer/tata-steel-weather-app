import os
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_historical_data(start_date="2018-01-01"):  # Changed default start_date
    """Fetch historical weather data from Open-Meteo API"""
    try:
        logging.info(f"Fetching historical data from {start_date} to today")
        
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        # Get today's date in YYYY-MM-DD format
        today = datetime.now().strftime("%Y-%m-%d")
        
        logging.info(f"Using coordinates: latitude=52.4779, longitude=4.61")

        # Make sure we use exactly the same parameters as in forecast
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 52.4779,
            "longitude": 4.61,
            "start_date": start_date,  # Will now start from 2017
            "end_date": today,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "apparent_temperature",
                "pressure_msl",
                "surface_pressure",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m"
            ]
        }

        logging.info("Making API request...")
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        logging.info("Processing response data...")
        hourly = response.Hourly()
        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}

        # Process variables in the same order as they appear in params["hourly"]
        for idx, var_name in enumerate(params["hourly"]):
            hourly_data[var_name] = hourly.Variables(idx).ValuesAsNumpy()

        # Create DataFrame
        hourly_dataframe = pd.DataFrame(data = hourly_data)
        
        # Convert timezone-aware timestamps to timezone-naive
        hourly_dataframe['date'] = pd.to_datetime(hourly_dataframe['date']).dt.tz_localize(None)
        
        return hourly_dataframe

    except Exception as e:
        logging.error(f"Error in get_historical_data: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        df = get_historical_data()
        logging.info("Saving data to CSV...")
        
        # Define the correct output path
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csv-api')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'historical.csv')
        
        df.to_csv(output_path, index=False)
        logging.info(f"Data successfully saved to {output_path}")
        
        # Print some basic statistics
        logging.info("\nData Summary:")
        logging.info(f"Total rows: {len(df)}")
        logging.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logging.info("\nMissing values per column:")
        for column in df.columns:
            missing = df[column].isna().sum()
            if missing > 0:
                logging.info(f"{column}: {missing} missing values")
                
    except Exception as e:
        logging.error(f"Script execution failed: {str(e)}")
