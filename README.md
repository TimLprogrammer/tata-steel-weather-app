# Weather Forecast and Model Management App

## Description

This application provides an interface for viewing weather forecasts and managing the machine learning models that generate these predictions. The focus is on predicting energy-related metrics (such as heating and cooling) based on weather data.

The app is built with Streamlit and Python, and uses machine learning models (LightGBM) trained on historical weather data and fault reports.

## Key Features

*   **Current Predictions:** View the latest weather forecasts and the derived energy needs.
*   **Model Management:**
    *   Visualize model performance (metrics and plots).
    *   Compare different models.
*   **Data Update & Model Retraining:**
    *   Initiate the process to fetch the latest weather data.
    *   Initiate the process to retrain the prediction models with the most recent data.
*   **Explanation:** Gain insights into how the models work and which features are important.

## Project Structure

```
main_project/
├── data/                     # Scripts and data for data collection and processing
│   ├── csv-api/              # Raw data from API (historical, forecast)
│   ├── csv-daily/            # Daily aggregated weather data
│   ├── script/               # Scripts for data processing (e.g., classifie_vks.py, process_daily_weather.py)
│   └── update.py             # Main script for data update pipeline
├── predictions/              # Scripts and data for model training and predictions
│   ├── best-model/           # Saved trained models (.pkl)
│   │   ├── cooling/
│   │   └── heating/
│   ├── plots/                # Generated plots of model evaluations
│   │   ├── cooling/
│   │   └── heating/
│   ├── cooling.py            # Script for training cooling model
│   ├── heating.py            # Script for training heating model
│   └── forecast.py           # Script to make predictions with trained models
├── streamlit/                # Streamlit application code
│   └── main.py               # Main file for the Streamlit UI
├── requirements.txt          # List of Python dependencies
├── README.md                 # This file
└── LICENSE                   # License information
```

## Requirements

*   Python 3.8+ (recommended)
*   pip (Python package installer)

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd main_project
    ```

2.  **(Recommended) Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  Navigate to the root directory where you cloned or downloaded the `main_project` (e.g., `cd path/to/main_project`).
2.  Ensure your virtual environment is activated if you created one.
3.  Run the following command in your terminal:

```bash
streamlit run streamlit/main.py
```

The application will open in your default web browser (usually at `http://localhost:8501`).

## Using the Application

*   **Navigation:** Use the sidebar to navigate between the different sections: "Latest Predictions", "Retrain Models", "Model Management", and "Explanation".
*   **Data Update:** In the "Retrain Models" section, click the "Update Weather Data" button to fetch the latest weather data. This process may take some time.
*   **Model Retraining:** After updating the data, or whenever desired, click "Retrain Cooling Model" or "Retrain Heating Model" in the "Retrain Models" section to retrain the respective models. This may also take some time.

## Contributing

For Tata Steel employees who wish to contribute:

1.  Fork the repository (if hosted on a central Git server).
2.  Create a new branch for your feature or bugfix (`git checkout -b feature/your-feature`).
3.  Commit your changes (`git commit -am 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
