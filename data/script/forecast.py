import os
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_forecast_data():
    """Fetch forecast weather data from Open-Meteo API (today to 6 days ahead)"""
    try:
        logging.info("Fetching forecast data for today to 6 days ahead")

        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 52.4779,
            "longitude": 4.61,
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

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

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

        logging.info(f"Forecast data fetched: {len(hourly_dataframe)} records from {hourly_dataframe['date'].min()} to {hourly_dataframe['date'].max()}")

        return hourly_dataframe

    except Exception as e:
        logging.error(f"Error in get_forecast_data: {str(e)}")
        raise

def get_recent_historical_data():
    """Fetch historical weather data from yesterday to one month ago"""
    try:
        # Calculate date range: yesterday to one month ago
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        one_month_ago = today - timedelta(days=30)  # Approximately one month

        yesterday_str = yesterday.strftime("%Y-%m-%d")
        one_month_ago_str = one_month_ago.strftime("%Y-%m-%d")

        logging.info(f"Fetching historical data from {one_month_ago_str} to {yesterday_str}")

        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 52.4779,
            "longitude": 4.61,
            "start_date": one_month_ago_str,
            "end_date": yesterday_str,
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

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

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

        logging.info(f"Historical data fetched: {len(hourly_dataframe)} records from {hourly_dataframe['date'].min()} to {hourly_dataframe['date'].max()}")

        return hourly_dataframe

    except Exception as e:
        logging.error(f"Error in get_recent_historical_data: {str(e)}")
        raise

def combine_and_save_data():
    """Combine historical and forecast data and save to forecast.csv"""
    try:
        # Get historical data (past month to yesterday)
        historical_df = get_recent_historical_data()

        # Get forecast data (today to 6 days ahead)
        forecast_df = get_forecast_data()

        # Combine the dataframes
        combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)

        # Sort by date to ensure chronological order
        combined_df = combined_df.sort_values('date')

        # Remove any duplicates that might occur at the boundary
        combined_df = combined_df.drop_duplicates(subset=['date'])

        # Define the output path
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv-api')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'forecast.csv')

        # Save to CSV
        combined_df.to_csv(output_path, index=False)
        logging.info(f"Combined data saved to {output_path}")
        logging.info(f"Total records: {len(combined_df)}")
        logging.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")

        return combined_df

    except Exception as e:
        logging.error(f"Error in combine_and_save_data: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        combine_and_save_data()
    except Exception as e:
        logging.error(f"Script execution failed: {str(e)}")

