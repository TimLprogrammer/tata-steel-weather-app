import streamlit as st
import subprocess
import pandas as pd
from datetime import datetime, timedelta
import json
import os

st.set_page_config(page_title="Weather Forecast & Model Management", layout="wide")

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Latest Predictions", "Models", "Explanation"])

if menu == "Latest Predictions":
    st.title("Latest Predictions")
    st.write("Run the latest weather data processing and view forecasts.")

    if 'run_clicked' not in st.session_state:
        st.session_state['run_clicked'] = False

    if st.button("Run Forecast Pipeline"):
        with st.spinner("Running the forecast pipeline, please wait..."):
            scripts = [
                "data/script/forecast.py",
                "data/script/historical.py",
                "data/script/process_daily_weather.py",
                "predictions/forecast.py"
            ]
            for script in scripts:
                process = subprocess.Popen(
                    ["python3", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                process.communicate()

        st.success("Forecast pipeline completed.")
        st.session_state['run_clicked'] = True

    if st.session_state['run_clicked']:
        try:
            df = pd.read_csv("predictions/forecast_predictions.csv", parse_dates=["date"])
            today = datetime.now().date()
            end_date = today + timedelta(days=6)
            mask = (df["date"].dt.date >= today) & (df["date"].dt.date <= end_date)
            filtered = df.loc[mask]

            with open("predictions/best-model/cooling/best_model_metrics.json") as f:
                cooling_metrics = json.load(f)
            with open("predictions/best-model/heating/best_model_metrics.json") as f:
                heating_metrics = json.load(f)

            train_dates = {
                "Cooling": cooling_metrics.get("timestamp", "unknown"),
                "Heating": heating_metrics.get("timestamp", "unknown")
            }

            st.markdown(f"**Based on models trained on:**  \n"
                        f"Cooling: {train_dates['Cooling']}  \n"
                        f"Heating: {train_dates['Heating']}")

            st.subheader("Forecasts for the next 7 days")
            st.table(filtered.round(2).reset_index(drop=True))

            for label, metrics in {"Cooling": cooling_metrics, "Heating": heating_metrics}.items():
                st.markdown(f"### {label} Model: **{metrics.get('model_name', 'Unknown')}**")
                m = metrics.get("metrics", {})
                st.write({
                    "R²": m.get("R²"),
                    "MSE": m.get("MSE"),
                    "RMSE": m.get("RMSE"),
                    "MAE": m.get("MAE")
                })

            st.markdown("""
### How to interpret these metrics:

- **R² (Coefficient of Determination):**  
  Measures how well the model explains the variation in the actual data.  
  - **Range:** 0 to 1 (sometimes negative if the model is very poor).  
  - **Interpretation:**  
    - **1:** Perfect prediction.  
    - **0:** Model explains none of the variation.  
    - **Closer to 1 is better.**  
  - **Importance:** Gives an overall sense of how well the model fits, but does not tell you about the size of errors.

- **MSE (Mean Squared Error):**  
  The average of the squared differences between predicted and actual values.  
  - **Sensitive to large errors** (because errors are squared).  
  - **Lower is better.**  
  - **Units:** Squared units of the target variable, which can be hard to interpret directly.

- **RMSE (Root Mean Squared Error):**  
  The square root of MSE, so it is in the same units as the target variable.  
  - **Easier to interpret** than MSE.  
  - **Lower is better.**  
  - **Sensitive to large errors** (like MSE).  
  - **Often preferred** when large errors are particularly undesirable.

- **MAE (Mean Absolute Error):**  
  The average of the absolute differences between predicted and actual values.  
  - **Lower is better.**  
  - **Less sensitive to outliers** than MSE or RMSE.  
  - **Gives a straightforward average error size.**

### Which metrics matter most?

- **R²** tells you how well the model explains the data overall.  
- **RMSE** and **MAE** tell you how big the typical errors are, in the same units as your target.  
- **RMSE** penalizes large errors more heavily, so if big mistakes are costly, focus on RMSE.  
- **MAE** is more robust to outliers and easier to interpret as "average error."

### Summary:

- **Higher R²** (closer to 1) means a better fit.  
- **Lower MSE, RMSE, and MAE** mean smaller errors and better predictions.  
- Use **R²** for overall fit, **RMSE** if large errors matter, and **MAE** for average error size.

Together, these metrics provide a complete picture of model quality.
""")
        except Exception as e:
            st.error(f"Error loading forecast predictions or model info: {e}")
    else:
        st.info("Click **Run Forecast Pipeline** to generate and view the latest predictions.")

elif menu == "Models":
    st.title("Model Management")

    uploaded_file = st.file_uploader("Upload the fault notifications Excel file", type=["xlsx"])

    if uploaded_file is not None:
        save_path = "data/uploaded_data.xlsx"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully.")

    if st.button("Retrain Models"):
        log_placeholder = st.empty()
        logs = ""

        scripts = [
            "data/update.py",
            "predictions/cooling.py",
            "predictions/heating.py"
        ]

        for script in scripts:
            process = subprocess.Popen(
                ["python3", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    logs += line
                    log_placeholder.text(logs)

        st.success("Model retraining completed.")

    if st.button("Show Current Model"):
        import plotly.graph_objects as go

        try:
            with open("predictions/best-model/cooling/best_model_metrics.json") as f:
                cooling_metrics_data = json.load(f)
            all_metrics = cooling_metrics_data.get("all_metrics", {})
        except Exception as e:
            st.error(f"Error loading cooling model metrics for plots: {e}")
            all_metrics = {}

        st.subheader("Detailed Model Visualizations")

        plot_types = {
            "metrics_comparison.png": """
**What do you see here?**  
A comparison of key error metrics (MSE, RMSE, MAE, R²) across different models.

**How to read this?**  
Lower values for MSE, RMSE, and MAE indicate better performance. A higher R² (maximum 1) means the model explains more variance.

**Why is this important?**  
It helps identify which model performs best on various evaluation criteria.

**What does it tell you?**  
Provides insight into the accuracy and reliability of each model, guiding the selection of the best one.
""",
            "residuals_analysis.png": """
**What do you see here?**  
A visualization of the residuals, which are the differences between the predicted and actual values for each data point.

**How to read this?**  
Ideally, residuals should be randomly scattered around zero without any clear pattern.  
Look for:
- **Random cloud:** Indicates a well-fitted model.
- **Patterns, curves, or lines:** Systematic shapes such as curved lines, straight lines, or waves suggest the model is missing some relationship (non-linearity or missing variables).
- **Funnel shape (wider spread at higher values):** Indicates heteroscedasticity, meaning the error variance changes with the predicted value.
- **Clusters or gaps:** May reveal subgroups or missing variables.
- **Outliers:** Large residuals that may unduly influence the model.

**Why is this important?**  
Residual analysis helps diagnose if the model assumptions hold:
- Constant variance
- No systematic bias
- Correct functional form

**What does it tell you?**  
Whether the model is consistent and reliable across the data range, or if there are issues like bias, non-linearity, or heteroscedasticity that suggest the model could be improved.
""",
            "feature_importance.png": """
**What do you see here?**  
The relative importance of input variables for the model.

**How to read this?**  
Higher scores mean the feature has more influence on the predictions.

**Why is this important?**  
Helps understand which factors most affect the outcome.

**What does it tell you?**  
Provides insight into the drivers of the model and potential focus areas for data improvement.
""",
            "predictions_comparison.png": """
**What do you see here?**  
A scatter plot comparing the predicted values to the actual observed values.

**How to read this?**  
- The diagonal line represents perfect predictions (predicted = actual).
- Points close to this line indicate accurate predictions.
- Systematic deviations from the line suggest bias:
  - **Above the line:** Model tends to underestimate.
  - **Below the line:** Model tends to overestimate.
- Spread around the line shows the variability of prediction errors.
- Clusters or gaps may indicate subgroups or data issues.
- Outliers far from the line highlight large prediction errors.

**Why is this important?**  
This plot visually assesses:
- Overall accuracy
- Presence of bias (systematic over- or underestimation)
- Consistency across the range of values
- Potential data issues or outliers

**What does it tell you?**  
Whether the model reliably predicts across the full range of data, or if there are systematic errors or inconsistencies that need to be addressed.
"""
        }

        model_names = ["Cooling", "Heating"]
        for model in model_names:
            st.subheader(f"{model} Model Plots")
            base_path = f"predictions/plots/{model.lower()}/"
            for generic_filename_key, explanation in plot_types.items():
                name_part, ext_part = os.path.splitext(generic_filename_key)
                actual_plot_filename = f"{name_part}-{model.lower()}{ext_part}"
                plot_path = os.path.join(base_path, actual_plot_filename)
                
                expander_title_base = generic_filename_key.replace('.png', '').replace('_', ' ').title()
                with st.expander(f"{model} - {expander_title_base}"):
                    if os.path.exists(plot_path):
                        st.image(plot_path)
                        st.write(explanation)
                    else:
                        st.warning(f"Plot file '{actual_plot_filename}' not found in '{base_path}'. Please ensure it is generated and placed correctly.")

elif menu == "Explanation":
    st.title("Pipeline Explanation")

    tab_titles = [
        "Pipeline Overview",
        "Folders & Files Explained",
        "Model Training & Retraining",
        "Latest Predictions Process",
        "Design Choices & Rationale"
    ]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.header("Pipeline Overview")
        st.write("""
This project is an automated system for weather-based fault prediction, designed to be easy understandable and manageable.

**How it works, step by step:**

1. **Collect data:**  
   - Fetches weather forecasts and historical weather data from APIs.  
   - Loads fault notifications from Excel files.

2. **Process data:**  
   - Cleans the raw data.  
   - Adds useful features (like day of week, month, wind categories).  
   - Aggregates data to daily summaries.

3. **Classify faults:**  
   - Uses rules and machine learning to categorize fault notifications.

4. **Train models:**  
   - Trains multiple machine learning models to predict faults based on weather and time features.  
   - Optimizes models and combines them into ensembles for better accuracy.

5. **Generate forecasts:**  
   - Uses the trained models to predict future faults based on the latest weather forecasts.

6. **Visualize results:**  
   - Shows forecasts, model performance, and detailed plots in an interactive app.

7. **Retrain regularly:**  
   - Updates data and retrains models to stay accurate over time.

**All these steps are automated and can be triggered from the app interface.**
        """)

    with tabs[1]:
        st.header("Folders & Files Explained")
        st.write("""
**`data/script/`**  
- `forecast.py`: Fetches weather forecast data.  
- `historical.py`: Fetches historical weather data.  
- `process_daily_weather.py`: Cleans and enriches weather data.  
- `classifie_vks.py`: Classifies fault notifications.  
- `result.py`: Creates visualizations of fault data.

**`data/`**  
- `update.py`: Updates datasets and triggers retraining.  
- `csv-api/` and `csv-daily/`: Store raw and processed weather data.  
- `notifications/`: Contains fault notification Excel files.

**`predictions/`**  
- `forecast.py`: Generates fault forecasts using trained models.  
- `cooling.py` and `heating.py`: Train, optimize, evaluate, and save models.  
- `best-model/`: Stores the best models and their metrics.  
- `plots/`: Contains visualizations of model performance.

**`streamlit/`**  
- `main.py`: The app interface to run the pipeline, retrain models, and view results.

*Each script and folder has a clear, focused role to keep the system organized and maintainable.*
        """)

    with tabs[2]:
        st.header("Model Training & Retraining")
        st.write("""
**How model training works:**

- Data is cleaned and features are engineered.
- Outliers are handled to improve robustness.
- Multiple algorithms are trained: LightGBM, XGBoost, Gradient Boosting, Random Forest, etc.
- Hyperparameters are optimized using automated search.
- The best models are combined into ensembles (Voting, Stacking).
- Models are evaluated on accuracy metrics.
- The best performing models and their metrics are saved.

**Retraining process:**

- When you click "Retrain Models" in the app, it:  
  1. Updates datasets with new fault notifications.  
  2. Retrains all models with the latest data.  
  3. Optimizes and evaluates models again.  
  4. Saves the new best models and metrics.  
  5. Updates plots and explanations.

**Logging during retraining:**

- Shows progress of each script (data update, cooling model, heating model).
- Displays training status, optimization steps, and evaluation results.
- Helps you understand what is happening at each step.
- Confirms when retraining is complete.

*This ensures models stay accurate and up-to-date with minimal manual effort.*
        """)

    with tabs[3]:
        st.header("Latest Predictions Process")
        st.write("""
**What happens when you click "Run Forecast Pipeline":**

1. **Sequential script execution:**  
   The app runs four scripts, one after the other:  
   - `data/script/forecast.py`: Fetches the latest weather forecast data.  
   - `data/script/historical.py`: Fetches recent historical weather data.  
   - `data/script/process_daily_weather.py`: Cleans, enriches, and aggregates the combined data.  
   - `predictions/forecast.py`: Loads the trained models, preprocesses the data, and generates fault predictions.

2. **Data flow:**  
   - The first three scripts update the weather datasets.  
   - The last script uses these datasets and the saved models to generate new predictions.

3. **Saving results:**  
   - The predictions are saved in `predictions/forecast_predictions.csv`.  
   - This file contains the expected fault counts for the next 7 days.

4. **Displaying results:**  
   - The app loads this CSV and filters it to show only the next 7 days.  
   - It also loads the latest model info (name, metrics, training date) from saved JSON files.  
   - All this is displayed in a clear table with explanations.

5. **Why this approach:**  
   - Ensures predictions are always based on the latest data.  
   - Automates a complex multi-step process with one button click.  
   - Provides transparency by showing which models were used and how good they are.  
   - Makes it easy for users to get up-to-date forecasts without technical knowledge.

*This process guarantees fresh, accurate predictions with minimal effort.*
        """)

    with tabs[4]:
        st.header("Design Choices & Rationale")
        st.write("""
- **Modular scripts:** Easier to maintain, test, and extend.
- **Automated data updates:** Keeps models relevant with new data.
- **Multiple algorithms:** Increases robustness and accuracy.
- **Ensemble models:** Combine strengths of different algorithms.
- **Feature engineering:** Improves predictive power.
- **Interactive app:** Makes complex processes accessible to all users.
- **Clear logging:** Shows what happens during training and prediction.
- **Visualizations:** Help interpret results and model behavior.
*All choices aim to create a transparent, accurate, and user-friendly system.*
        """)
