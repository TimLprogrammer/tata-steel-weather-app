# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
import warnings
import optuna
from optuna.samplers import TPESampler
import os
import logging
import pickle
import json
from datetime import datetime
import os
from scipy import stats
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV

# Define base path using relative paths
import os.path as path

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console only
    ]
)

# Get the directory of the current script
SCRIPT_DIR = path.dirname(path.abspath(__file__))
# Get the main_project directory (parent of predictions directory)
BASE_PATH = path.dirname(SCRIPT_DIR)
logger.info(f"Script directory: {SCRIPT_DIR}")
logger.info(f"Base path: {BASE_PATH}")

# Set LightGBM logging level
logger = logging.getLogger('lightgbm')
logger.setLevel(logging.ERROR)

# Set LightGBM environment variable
os.environ['LIGHTGBM_VERBOSE'] = '-1'

# At the top of your file, add these specific warning filters
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', message='[LightGBM]')

# Function for detection and handling of outliers
def handle_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Handle outliers in the dataset

    Parameters:
        df (DataFrame): Input dataframe
        columns (list): Columns to check for outliers (None = all numeric)
        method (str): 'iqr' or 'zscore'
        threshold (float): Threshold for outlier detection

    Returns:
        DataFrame: Dataframe without outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns

    df_clean = df.copy()

    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Use winsorization instead of removing
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

        elif method == 'zscore':
            z_scores = stats.zscore(df[col])
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < threshold)

            # Use winsorization
            df_clean.loc[~filtered_entries, col] = np.nan
            df_clean[col] = df_clean[col].fillna(df[col].median())

    return df_clean

# Function for data processing and feature engineering
def preprocess_data(df, target_col='fault_count'):
    """
    Preprocess the data and create features, excluding the target column from feature engineering
    """
    df = df.copy()

    # Convert date column if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Detect and handle outliers in numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    target_mask = numeric_cols != target_col
    cols_for_outlier_handling = numeric_cols[target_mask]
    df = handle_outliers(df, columns=cols_for_outlier_handling)

    # Extract date features
    if 'date' in df.columns:
        df['dag'] = df['date'].dt.day
        df['week'] = df['date'].dt.isocalendar().week
        df['maand'] = df['date'].dt.month
        df['jaar'] = df['date'].dt.year
        df['weekdag'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['weekdag'].apply(lambda x: 1 if x >= 5 else 0)
        df['kwartaal'] = df['date'].dt.quarter
        df['dag_van_jaar'] = df['date'].dt.dayofyear
        df['week_van_jaar'] = df['date'].dt.isocalendar().week

    # Add season if not present
    if 'seizoen' not in df.columns:
        df['seizoen'] = df['maand'].apply(lambda x: 'winter' if x in [12, 1, 2] else
                                  'lente' if x in [3, 4, 5] else
                                  'zomer' if x in [6, 7, 8] else 'herfst')

    # Create derived temperature features if they exist
    if 'temperatuur_min' in df.columns and 'temperatuur_max' in df.columns:
        df['temp_range'] = df['temperatuur_max'] - df['temperatuur_min']

    if 'temperatuur_avg' in df.columns:
        # Rolling statistics over temperature (7-day window)
        df['temp_avg_7d_mean'] = df['temperatuur_avg'].rolling(window=7, min_periods=1).mean()
        df['temp_avg_7d_std'] = df['temperatuur_avg'].rolling(window=7, min_periods=1).std().fillna(0)
        df['temp_avg_7d_min'] = df['temperatuur_avg'].rolling(window=7, min_periods=1).min()
        df['temp_avg_7d_max'] = df['temperatuur_avg'].rolling(window=7, min_periods=1).max()

        # Temperature variation indicators
        df['temp_change'] = df['temperatuur_avg'].diff().fillna(0)
        df['temp_acc'] = df['temp_change'].diff().fillna(0)  # Acceleration in temperature change

        # Extreme temperature indicators
        df['extreme_temp'] = ((df['temperatuur_avg'] > df['temperatuur_avg'].quantile(0.95)) |
                               (df['temperatuur_avg'] < df['temperatuur_avg'].quantile(0.05))).astype(int)

    # Remove any lag features or other features derived from target column
    columns_to_drop = [col for col in df.columns if target_col in col.lower() and col != target_col]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    return df

# Function for feature selection
def select_features(df, target_col='fault_count', correlation_threshold=0.95, variance_threshold=1e-3):
    """
    Select relevant features by removing low variance and highly correlated features.
    Parameters:
        df (DataFrame): Dataset with features
        target_col (str): Name of the target variable column
        correlation_threshold (float): Threshold for removing correlated features
        variance_threshold (float): Threshold for removing features with low variance
    Returns:
        DataFrame: Dataset with selected features
    """
    # Separate features and target
    if target_col in df.columns:
        features = df.drop(columns=[target_col])
        if 'date' in features.columns:
            features = features.drop(columns=['date'])
    else:
        features = df.copy()
        if 'date' in features.columns:
            features = features.drop(columns=['date'])

    # Remove features with low variance
    low_variance_cols = [col for col in features.columns if features[col].std() < variance_threshold]
    if low_variance_cols:
        logger.info(f"Removing {len(low_variance_cols)} features with low variance")
        features = features.drop(columns=low_variance_cols)

    # Remove highly correlated features
    corr_matrix = features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    if to_drop:
        logger.info(f"Removing {len(to_drop)} highly correlated features")
        features = features.drop(columns=to_drop)

    # Calculate correlation with target (if available) for feature ranking
    if target_col in df.columns:
        target_correlations = features.apply(lambda x: x.corr(df[target_col]))
        # Print top 10 features by correlation
        logger.info("Top 10 features by correlation with target:")
        logger.info(f"\n{target_correlations.abs().sort_values(ascending=False).head(10).to_string()}")

    return features

# Function for hyperparameter optimization with Optuna
def optimize_lightgbm(X_train, y_train, n_trials=20):
    """
    Optimize LightGBM hyperparameters with Optuna.

    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : Series or array
        Target labels
    n_trials : int
        Number of optimization iterations

    Returns:
    --------
    dict
        Optimal hyperparameters
    """
    # Create validation set for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    eval_set = [(X_val, y_val)]

    def objective(trial):
        # Define the parameter space
        params = {
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'objective': 'regression',
            'verbosity': -1,
            'random_state': 42
        }

        # Create and train the model
        model = lgb.LGBMRegressor(**params)

        # FIX: use callbacks for early_stopping instead of as an argument
        callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False)]

        model.fit(
            X_tr, y_tr,
            eval_set=eval_set,
            callbacks=callbacks
        )

        # Predict and evaluate
        preds = model.predict(X_val)
        r2 = r2_score(y_val, preds)

        return r2

    # Create and run the study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best R² score: {study.best_value:.4f}")
    logger.info(f"Best parameters:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")

    # Add extra parameters that are not optimized
    best_params = study.best_params.copy()
    best_params['objective'] = 'regression'
    best_params['verbosity'] = -1
    best_params['random_state'] = 42

    return best_params

# Function for XGBoost hyperparameter optimization
def optimize_xgboost(X, y, n_trials=15):
    """
    Optimize XGBoost hyperparameters with Optuna.
    """
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }

        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=5)

        # Cross-validation
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBRegressor(**param, random_state=42, objective='reg:squarederror')
            # Fix: use callbacks instead of early_stopping_rounds
            eval_set = [(X_val_cv, y_val_cv)]
            model.fit(X_train_cv, y_train_cv,
                     eval_set=eval_set,
                     verbose=False)

            y_pred = model.predict(X_val_cv)
            r2 = r2_score(y_val_cv, y_pred)
            cv_scores.append(r2)

        return np.mean(cv_scores)  # We maximize R²

    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Add random_state to the best parameters
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['objective'] = 'reg:squarederror'

    logger.info(f"Best XGBoost parameters: {best_params}")
    logger.info(f"Best R² score: {study.best_value:.4f}")

    return best_params

# Improved function for GradientBoosting optimization
def optimize_gradient_boosting(X, y, n_trials=100):
    """
    Optimize GradientBoosting hyperparameters with Optuna.
    """
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'alpha': trial.suggest_float('alpha', 0.1, 0.99),
        }

        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=5)

        # Cross-validation
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

            model = GradientBoostingRegressor(**param, random_state=42)
            model.fit(X_train_cv, y_train_cv)

            y_pred = model.predict(X_val_cv)
            rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred))
            cv_scores.append(rmse)

        return np.mean(cv_scores)

    study = optuna.create_study(direction='minimize', sampler=TPESampler())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Add random_state to the best parameters
    best_params = study.best_params
    best_params['random_state'] = 42

    logger.info(f"Best GradientBoosting parameters: {best_params}")
    logger.info(f"Best RMSE score: {study.best_value:.4f}")

    return best_params

# Function for training an ensemble model
def train_ensemble_model(X_train, y_train, X_test, y_test, lgbm_params=None, gb_params=None, xgb_params=None):
    """Train various models and create a sophisticated ensemble"""
    models = {}
    base_predictions = {}
    base_metrics = {}

    # Add base parameters for LightGBM
    lgbm_params = lgbm_params or {}
    lgbm_params.update({
        'verbose': -1,
        'verbosity': -1,
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt'
    })

    # Default GradientBoosting parameters if not optimized
    gb_params = gb_params or {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'subsample': 0.8,
        'random_state': 42
    }

    # Default XGBoost parameters if not optimized
    xgb_params = xgb_params or {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 3,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.01,
        'reg_lambda': 1.0,
        'random_state': 42,
        'objective': 'reg:squarederror'
    }

    # Create the base models
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        ),
        'GradientBoosting': GradientBoostingRegressor(**gb_params),
        'LightGBM': lgb.LGBMRegressor(**lgbm_params),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            random_state=42,
            verbosity=0,
            objective='reg:squarederror'
        ),
        'AdaBoost': AdaBoostRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),
        'ElasticNet': ElasticNet(
            alpha=0.5,
            l1_ratio=0.5,
            max_iter=10000,
            random_state=42
        ),
        'SVR': SVR(
            kernel='rbf',
            C=10,
            epsilon=0.1,
            gamma='scale'
        )
    }

    # Train base models and collect predictions
    for name, model in models.items():
        logger.info(f"Training {name} model...")
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        base_predictions[name] = y_pred

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        base_metrics[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
        print(f"{name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    # Create two types of ensembles - Stacking and Voting

    # 1. Voting Regressor (weighted)
    estimators = [(name, model) for name, model in models.items()]
    # Weighted based on R² scores
    weights = [max(base_metrics[name]['R²'], 0.01) for name, _ in estimators]
    # Normalize weights
    weights = [w/sum(weights) for w in weights]

    logger.info("Training Weighted Voting Ensemble...")
    voting_regressor = VotingRegressor(
        estimators=estimators,
        weights=weights
    )
    voting_regressor.fit(X_train, y_train)
    voting_pred = voting_regressor.predict(X_test)

    # 2. Multi-level Stacking ensemble
    logger.info("Training Multi-level Stacking Ensemble...")

    # Select only the best models for the ensemble - use all models for better performance
    best_models = []
    for name, model in models.items():
        if base_metrics[name]['R²'] > 0.3:  # Lower threshold to include more models
            best_models.append((name, model))

    logger.info(f"Selected models for stacking: {[name for name, _ in best_models]}")

    # Level 1 regressor - use XGBoost as meta-regressor (best model)
    meta_regressor = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )

    # Use KFold with shuffle=False to better respect time series nature
    # This is a compromise since TimeSeriesSplit doesn't work with cross_val_predict
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=5, shuffle=False)
    stacking_model = StackingRegressor(
        estimators=best_models,
        final_estimator=meta_regressor,
        cv=kfold,
        n_jobs=-1
    )

    stacking_model.fit(X_train, y_train)
    stacking_pred = stacking_model.predict(X_test)

    # Evaluate both ensembles

    # Voting ensemble metrics
    voting_mse = mean_squared_error(y_test, voting_pred)
    voting_rmse = np.sqrt(voting_mse)
    voting_mae = mean_absolute_error(y_test, voting_pred)
    voting_r2 = r2_score(y_test, voting_pred)

    print(f"Voting Ensemble - MSE: {voting_mse:.4f}, RMSE: {voting_rmse:.4f}, MAE: {voting_mae:.4f}, R²: {voting_r2:.4f}")

    # Stacking ensemble metrics
    stacking_mse = mean_squared_error(y_test, stacking_pred)
    stacking_rmse = np.sqrt(stacking_mse)
    stacking_mae = mean_absolute_error(y_test, stacking_pred)
    stacking_r2 = r2_score(y_test, stacking_pred)

    print(f"Stacking Ensemble - MSE: {stacking_mse:.4f}, RMSE: {stacking_rmse:.4f}, MAE: {stacking_mae:.4f}, R²: {stacking_r2:.4f}")

    # Add ensembles to models and metrics
    models['VotingEnsemble'] = voting_regressor
    models['StackingEnsemble'] = stacking_model

    base_predictions['VotingEnsemble'] = voting_pred
    base_predictions['StackingEnsemble'] = stacking_pred

    # Add ensemble metrics
    base_metrics['VotingEnsemble'] = {
        'MSE': voting_mse,
        'RMSE': voting_rmse,
        'MAE': voting_mae,
        'R²': voting_r2
    }

    base_metrics['StackingEnsemble'] = {
        'MSE': stacking_mse,
        'RMSE': stacking_rmse,
        'MAE': stacking_mae,
        'R²': stacking_r2
    }

    #base_metrics['VotingEnsemble'] = {
    #    'MSE': voting_mse,
    #    'RMSE': voting_rmse,
    #    'MAE': voting_mae,
    #    'R²': voting_r2
    #}

    #base_metrics['StackingEnsemble'] = {
    #    'MSE': stacking_mse,
    #    'RMSE': stacking_rmse,
    #    'MAE': stacking_mae,
    #    'R²': stacking_r2
    #}

    ## Choose the best model as the final ensemble
    #if voting_r2 > stacking_r2:
    #    final_ensemble = voting_regressor
    #    print("\nVoting Ensemble is better than Stacking Ensemble")
    #else:
    #    final_ensemble = stacking_model
    #    print("\nStacking Ensemble is better than Voting Ensemble")

    # Choose the best model as the final ensemble
    if voting_r2 > stacking_r2:
        final_ensemble = voting_regressor
        logger.info("Voting Ensemble is better than Stacking Ensemble")
    else:
        final_ensemble = stacking_model
        logger.info("Stacking Ensemble is better than Voting Ensemble")

    return final_ensemble, models, base_predictions, base_metrics

# Function for visualizing the results
def visualize_results(y_test, predictions, metrics, save_path):
    """
    Visualize the results of different models and save plots.

    Parameters:
    y_test: Actual values
    predictions (dict): Dictionary with model predictions
    metrics (dict): Dictionary with model metrics
    save_path (str): Path where plots are saved
    """
    # Create plot directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Plots are being saved in: {save_path}")

    # Plot actual vs. predicted values
    plt.figure(figsize=(20, 12))

    num_plots = min(len(predictions), 6)  # Limit to 6 plots

    for i, (name, y_pred) in enumerate(predictions.items()):
        if i >= num_plots:
            break
        plt.subplot(2, 3, i+1)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(f"{name} - Werkelijk vs. Voorspeld (R² = {metrics[name]['R²']:.4f})")
        plt.xlabel('Werkelijk')
        plt.ylabel('Voorspeld')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'predictions_comparison.png'))
    plt.close()

    # Compare model performances with barplot
    plt.figure(figsize=(15, 10))

    # Compare metrics
    metrics_plots = {
        'MSE': ([metrics[model]['MSE'] for model in metrics], 'Mean Squared Error (MSE)'),
        'RMSE': ([metrics[model]['RMSE'] for model in metrics], 'Root Mean Squared Error (RMSE)'),
        'MAE': ([metrics[model]['MAE'] for model in metrics], 'Mean Absolute Error (MAE)'),
        'R²': ([metrics[model]['R²'] for model in metrics], 'R² Score')
    }

    for i, (metric, (values, title)) in enumerate(metrics_plots.items(), 1):
        plt.subplot(2, 2, i)
        plt.bar(metrics.keys(), values)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics_comparison.png'))
    plt.close()

    # Plot residuals
    plt.figure(figsize=(20, 15))
    for i, (name, y_pred) in enumerate(predictions.items(), 1):
        residuals = y_test - y_pred
        plt.subplot(3, 3, i)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f"{name} - Residual Analysis")
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'residuals_analysis.png'))
    plt.close()

# Feature importance visualization with shap values if possible
def visualize_feature_importance(models, feature_names, save_path):
    """
    Visualize feature importance for different models and save.
    Parameters:
        models (dict): Dictionary with trained models
        feature_names (list): List with feature names
        save_path (str): Path where plots are saved
    """
    plt.figure(figsize=(20, 15))

    # Try to display feature importance for each model
    plot_count = 0
    for name, model in models.items():
        # Skip models without feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            continue

        plot_count += 1
        plt.subplot(3, 2, plot_count)

        try:
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            top_n = 15
            top_indices = indices[:top_n]
            top_importances = importances[top_indices]
            top_features = [feature_names[i] for i in top_indices]

            plt.barh(range(len(top_features)), top_importances, align='center')
            plt.yticks(range(len(top_features)), top_features)
            plt.title(f"Top {top_n} Feature Importance - {name}")
            plt.xlabel('Importance')
        except Exception as e:
            logger.error(f"Error plotting feature importance for {name}: {e}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_importance.png'))
    plt.close()

def save_best_model(models, metrics, save_path):
    """
    Save the best model based on R² score.

    Parameters:
    models (dict): Dictionary with all models
    metrics (dict): Dictionary with metrics per model
    save_path (str): Path where model is saved
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Model directory created/checked: {save_path}")

    # Find the best model based on R² score
    best_model_name = max(metrics.items(), key=lambda x: x[1]['R²'])[0]

    # Get the best model (either from base models or use ensemble if it's the best)
    if best_model_name in ['VotingEnsemble', 'StackingEnsemble']:
        best_model = models[best_model_name]
    else:
        best_model = models[best_model_name]

    best_metrics = metrics[best_model_name]

    # Save model
    model_path = os.path.join(save_path, f'best_model_heating.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    # Also save all models for ensemble use
    all_models_path = os.path.join(save_path, 'all_models_heating.pkl')
    with open(all_models_path, 'wb') as f:
        pickle.dump(models, f)

    # Save metrics
    metrics_path = os.path.join(save_path, 'best_model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'model_name': best_model_name,
            'metrics': best_metrics,
            'all_metrics': {k: v for k, v in metrics.items()},
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=4)

    logger.info(f"Best model ({best_model_name}) saved in: {model_path}")
    logger.info(f"Metrics: R²={best_metrics['R²']:.4f}, RMSE={best_metrics['RMSE']:.4f}")

def add_temporal_features(df):
    """Add time-dependent features to the dataset, simplified to match heating_backup.py"""
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()

    # 1. Cyclical time features (same as in heating_backup.py)
    if 'maand' in df.columns:
        df['jaar_sin'] = np.sin(2 * np.pi * df['maand'] / 12)
        df['jaar_cos'] = np.cos(2 * np.pi * df['maand'] / 12)
    if 'weekdag' in df.columns:
        df['week_sin'] = np.sin(2 * np.pi * df['weekdag'] / 7)
        df['week_cos'] = np.cos(2 * np.pi * df['weekdag'] / 7)

    # 2. Temperature and humidity seasonal interactions (same as in heating_backup.py)
    if 'temperatuur_avg' in df.columns:
        df['temp_seizoen'] = df['temperatuur_avg'] * df['jaar_sin']
    if 'luchtvochtigheid_avg' in df.columns:
        df['vocht_seizoen'] = df['luchtvochtigheid_avg'] * df['jaar_sin']

    # 3. Season specific temperature trends (same as in heating_backup.py)
    if 'temperatuur_avg' in df.columns and 'seizoen' in df.columns:
        for seizoen in df['seizoen'].unique():
            mask = df['seizoen'] == seizoen
            df.loc[mask, f'temp_trend_{seizoen}'] = df.loc[mask, 'temperatuur_avg'].diff()

    # Fill missing values
    return df.fillna(0)

def perform_cross_validation(models, X, y, cv=5):
    """
    Perform cross-validation for all models
    """
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        cv_results[name] = {
            'mean_r2': scores.mean(),
            'std_r2': scores.std()
        }
        logger.info(f"{name} CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    return cv_results

# Main function
def main():
    # Load data with relative path
    try:
        data_path = path.join(BASE_PATH, 'data', 'csv-daily', 'heating', 'heating_daily.csv')
        logger.info(f"Trying to load data from: {data_path}")
        data = pd.read_csv(data_path)
        logger.info(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    except FileNotFoundError:
        # Try alternative path as fallback
        try:
            alt_data_path = path.join(BASE_PATH, 'data', 'heating_daily.csv')
            logger.warning(f"First path not found, trying alternative path: {alt_data_path}")
            data = pd.read_csv(alt_data_path)
            logger.info(f"Dataset loaded from alternative path: {data.shape[0]} rows, {data.shape[1]} columns")
        except Exception as e:
            logger.error(f"Error loading the data: {e}")
            return
    except Exception as e:
        logger.error(f"Error loading the data: {e}")
        return

    # Log the first few rows for verification
    logger.info("Example of loaded data:")
    logger.info(f"\n{data.head().to_string()}")

    # Check dataset integrity
    logger.info("Dataset Information:")
    logger.info(f"Number of rows: {data.shape[0]}")
    logger.info("Missing values per column:")
    logger.info(f"\n{data.isnull().sum().to_string()}")

    # Assume that 'fault_count' is the target variable
    target_col = 'fault_count'

    if target_col not in data.columns:
        logger.error(f"Error: Target column '{target_col}' not found in dataset.")
        return

    # First preprocess data to create basic features including weekdag
    logger.info("Performing data preprocessing and feature engineering...")
    processed_data = preprocess_data(data, target_col='fault_count')
    logger.info(f"After preprocessing: {processed_data.shape[0]} rows, {processed_data.shape[1]} columns")

    # Then add temporal features that depend on weekdag
    logger.info("Adding temporal features...")
    processed_data = add_temporal_features(processed_data)

    # Feature selection
    logger.info("Performing feature selection...")
    features = processed_data.drop(columns=['fault_count', 'date'], errors='ignore')
    target = processed_data['fault_count']

    logger.info(f"After feature selection: {features.shape[1]} features retained")

    # For heating data we use chronological split since it performed better in heating_backup.py
    # This is more appropriate for time series data
    use_time_split = True  # Use chronological split for time series data

    if use_time_split and 'date' in processed_data.columns:
        logger.info("Using chronological data splitting because dataset contains time series...")
        processed_data = processed_data.sort_values('date')
        train_size = int(len(processed_data) * 0.8)
        X_train = features.iloc[:train_size]
        X_test = features.iloc[train_size:]
        y_train = target.iloc[:train_size]
        y_test = target.iloc[train_size:]
    else:
        logger.info("Using regular random split...")
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # Feature scaling - Use RobustScaler as in the backup version which performed better
    logger.info("Performing feature scaling...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # Hyperparameter optimization for LightGBM with Optuna
    logger.info("Performing hyperparameter optimization for LightGBM...")
    try:
        lgbm_params = optimize_lightgbm(X_train_scaled, y_train, n_trials=20)
    except Exception as e:
        logger.error(f"Error during hyperparameter optimization: {e}")
        lgbm_params = None

    # Accelerated hyperparameter optimization for XGBoost (the best model)
    logger.info("Performing XGBoost hyperparameter optimization...")
    try:
        # Optimize XGBoost parameters with fewer trials for faster execution
        xgb_params = optimize_xgboost(X_train_scaled, y_train, n_trials=5)
    except Exception as e:
        logger.error(f"Error during XGBoost optimization: {e}")
        # Use predefined parameters that work well for XGBoost
        xgb_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 3,
            'gamma': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 1.0,
            'random_state': 42,
            'objective': 'reg:squarederror'
        }

    # Optimize GradientBoosting hyperparameters
    logger.info("Performing GradientBoosting hyperparameter optimization...")
    try:
        # Use more trials (20) for better optimization as in the backup version
        gb_params = optimize_gradient_boosting(X_train_scaled, y_train, n_trials=20)
    except Exception as e:
        logger.error(f"Error during GradientBoosting optimization: {e}")
        # Use predefined parameters that worked well in heating_backup.py
        gb_params = {
            'n_estimators': 752,
            'learning_rate': 0.0042817383866057,
            'max_depth': 3,
            'min_samples_split': 9,
            'min_samples_leaf': 7,
            'max_features': 0.4370271798422256,
            'subsample': 0.8492844823039463,
            'alpha': 0.24169245072202689,
            'random_state': 42
        }

    # Train ensemble model
    logger.info("Training ensemble model...")
    ensemble_model, base_models, predictions, metrics = train_ensemble_model(
        X_train_scaled, y_train, X_test_scaled, y_test, lgbm_params, gb_params, xgb_params
    )

    # Visualize results
    logger.info("Visualizing the results...")
    # Define plots path within the function for better visibility
    plots_path = path.join(BASE_PATH, 'predictions', 'plots', 'heating')
    os.makedirs(plots_path, exist_ok=True)
    logger.info(f"Plots are being saved in: {plots_path}")
    visualize_results(y_test, predictions, metrics, plots_path)

    # Visualize feature importance
    logger.info("Visualizing feature importance...")
    visualize_feature_importance(base_models, X_train.columns, plots_path)

    # Save best model
    logger.info("Saving best model...")
    # Define models path within the function for better visibility
    best_models_path = path.join(BASE_PATH, 'predictions', 'best-model', 'heating')
    os.makedirs(best_models_path, exist_ok=True)
    logger.info(f"Models are being saved in: {best_models_path}")
    # Add ensemble model to the models dictionary before saving
    base_models['Ensemble'] = ensemble_model
    save_best_model(base_models, metrics, best_models_path)

    logger.info("Analysis completed!")

    # Print the summary of results
    summary_header = "="*50
    logger.info(f"\n{summary_header}")
    logger.info("SUMMARY OF RESULTS")
    logger.info(f"{summary_header}")
    best_model_name = max(metrics.items(), key=lambda x: x[1]['R²'])[0]
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"R² Score: {metrics[best_model_name]['R²']:.4f}")
    logger.info(f"RMSE: {metrics[best_model_name]['RMSE']:.4f}")
    logger.info(f"MAE: {metrics[best_model_name]['MAE']:.4f}")
    logger.info(f"MSE: {metrics[best_model_name]['MSE']:.4f}")
    logger.info(f"{summary_header}")

    # Feature importance for the best model (print)
    if hasattr(base_models[best_model_name], 'feature_importances_'):
        logger.info("Top 10 most important features:")
        importances = base_models[best_model_name].feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(10, X_train.shape[1])):
            logger.info(f"{i+1}. {X_train.columns[indices[i]]}: {importances[indices[i]]:.4f}")

    return ensemble_model, base_models, features, scaler

# Function to print the top features
def print_top_features(model, feature_names):
    """
    Print the top 10 most important features of a model.

    Parameters:
    model (object): A trained model with a feature_importances_ attribute.
    feature_names (list): A list with the names of the features.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        logger.info("Top 10 most important features:")
        for i in range(min(10, len(feature_names))):
            logger.info(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    else:
        logger.info("Feature importances are not available for this model.")

if __name__ == "__main__":
    # Configure root logger when running as main script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Output to console only
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting heating model training and evaluation")

    ensemble_model, base_models, features, scaler = main()

    # Print the top features of the best model
    if ensemble_model is not None:
        logger.info("Displaying top features of the ensemble model:")
        print_top_features(ensemble_model, features.columns)
    else:
        logger.warning("No ensemble model was created, cannot display top features")

    logger.info("Heating model training and evaluation completed")
