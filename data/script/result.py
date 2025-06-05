import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_fault_data(notifications_file, system_type):
    """Process fault notifications and return daily counts."""
    try:
        # Read notifications data
        df = pd.read_excel(notifications_file)
        system_name = 'cooling' if system_type == 'SF-VKS01' else 'heating'
        
        # Filter for specific system
        system_df = df[df['Verantw.werkpl.'] == system_type].copy()
        
        # Convert date column
        system_df['Gecreëerd op'] = pd.to_datetime(system_df['Gecreëerd op'])
        system_df['date'] = system_df['Gecreëerd op'].dt.date
        
        # Group by date and count faults
        daily_faults = system_df.groupby('date').size().reset_index()
        daily_faults.columns = ['date', 'fault_count']
        
        # Convert date to datetime for merging
        daily_faults['date'] = pd.to_datetime(daily_faults['date'])
        
        return daily_faults
    except Exception as e:
        logging.error(f"Error processing fault data: {e}")
        return None

def create_fault_boxplot(daily_data, system_name, output_dir):
    """Create boxplot for daily fault counts and detect outliers."""
    try:
        # Create boxplot directory if it doesn't exist
        boxplot_dir = os.path.join(output_dir, 'boxplot')
        os.makedirs(boxplot_dir, exist_ok=True)
        
        # Create boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=daily_data['fault_count'])
        plt.title(f'Boxplot Daily Faults - {system_name}')
        plt.ylabel('Number of Faults per Day')
        plt.tight_layout()
        
        # Save boxplot
        plt.savefig(os.path.join(boxplot_dir, f'faults_boxplot_{system_name}.png'))
        plt.close()
        
        # Detect and remove outliers using z-score
        z_scores = np.abs(stats.zscore(daily_data['fault_count']))
        outliers = daily_data[z_scores > 6]  # Verhoogd van 4 naar 6
        if not outliers.empty:
            logging.info(f"\nOutliers found for {system_name}:")
            for _, row in outliers.iterrows():
                logging.info(f"Date: {row['date']}, Fault count: {row['fault_count']}")
        
        # Remove outliers
        daily_data = daily_data[z_scores <= 6]  # Verhoogd van 4 naar 6
        return daily_data
    except Exception as e:
        logging.error(f"Error creating boxplot: {e}")
        return daily_data

def main():
    # Define paths
    base_path = os.path.join(".", "main_project")
    notifications_file = os.path.join(base_path, "data/notifications/vks_classified.xlsx")  # Updated filename
    historical_weather = os.path.join(base_path, "data/csv-daily/historical_daily.csv")
    output_base = os.path.join(base_path, "data/csv-daily")

    # Create output directories
    cooling_dir = os.path.join(output_base, "cooling")
    heating_dir = os.path.join(output_base, "heating")
    os.makedirs(cooling_dir, exist_ok=True)
    os.makedirs(heating_dir, exist_ok=True)

    # Process weather data
    logging.info("Loading historical weather data...")
    weather_df = pd.read_csv(historical_weather)
    weather_df['date'] = pd.to_datetime(weather_df['date'])

    # Process each system type
    systems = {
        'SF-VKS01': ('cooling', cooling_dir),
        'SF-VKS02': ('heating', heating_dir)
    }

    for system_type, (system_name, output_dir) in systems.items():
        logging.info(f"\nProcessing {system_name} data...")
        
        # Get fault data
        daily_faults = process_fault_data(notifications_file, system_type)
        if daily_faults is None:
            continue
            
        # Remove outliers from fault data
        daily_faults = create_fault_boxplot(daily_faults, system_name, output_dir)
        
        # Merge weather and fault data using inner join to keep only matching dates
        result_df = pd.merge(weather_df, daily_faults, on='date', how='inner')
        
        # Keep date column and reorder other columns
        cols = ['date', 'fault_count'] + [col for col in result_df.columns if col not in ['fault_count', 'date']]
        result_df = result_df[cols]
        
        # Save to CSV
        output_file = os.path.join(output_dir, f"{system_name}_daily.csv")
        result_df.to_csv(output_file, index=False)
        logging.info(f"Saved {system_name} data to {output_file}")
        logging.info(f"Final shape: {result_df.shape}")

if __name__ == "__main__":
    main()
