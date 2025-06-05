import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import logging
import re
import os


# Configure logging - without file handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Only keep console output
    ]
)
logger = logging.getLogger(__name__)


# Get notifications path from environment variable or use default
notifications_path = os.environ.get('NOTIFICATIONS_PATH')
if not notifications_path:
    # Use default path relative to the script location
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    notifications_path = os.path.join(base_dir, 'data', 'notifications', 'fm_vks.xlsx')
    logger.info(f"NOTIFICATIONS_PATH not set, using default path: {notifications_path}")

# Verify the file exists
if not os.path.exists(notifications_path):
    raise FileNotFoundError(f"Notifications file not found at: {notifications_path}")


def load_data(file_path):
    """Load dataset from Excel file"""
    logger.info(f"Loading data from {file_path}")
    return pd.read_excel(file_path, sheet_name='Blad1')


def train_binary_classifier(df, vks_category, features_cols=['Omschrijving', 'Omschrijving11']):
    """Train a binary classifier for a specific VKS category with k-fold cross validation"""
    logger.info(f"Training binary classifier for {vks_category}")
    
    # Create target variable
    df['target'] = df['Verantw.werkpl.'].apply(lambda x: 1 if x == vks_category else 0)
    
    # Combine text features
    df['combined_text'] = df[features_cols].apply(
        lambda row: ' '.join([str(val) if not pd.isna(val) else '' for val in row]), 
        axis=1
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Perform k-fold cross validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df['combined_text'], df['target'])):
        X_train = df['combined_text'].iloc[train_idx]
        X_val = df['combined_text'].iloc[val_idx]
        y_train = df['target'].iloc[train_idx]
        y_val = df['target'].iloc[val_idx]
        
        # Train model
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        
        # Calculate metrics
        cv_scores['precision'].append(precision_score(y_val, y_pred))
        cv_scores['recall'].append(recall_score(y_val, y_pred))
        cv_scores['f1'].append(f1_score(y_val, y_pred))
    
    # Log cross-validation results
    logger.info(f"Cross-validation results for {vks_category}:")
    logger.info(f"Precision: {np.mean(cv_scores['precision']):.3f} (+/- {np.std(cv_scores['precision']):.3f})")
    logger.info(f"Recall: {np.mean(cv_scores['recall']):.3f} (+/- {np.std(cv_scores['recall']):.3f})")
    logger.info(f"F1: {np.mean(cv_scores['f1']):.3f} (+/- {np.std(cv_scores['f1']):.3f})")
    
    # Train final model on full training data
    pipeline.fit(df['combined_text'], df['target'])
    return pipeline, cv_scores


def apply_keyword_rules(row):
    """Apply keyword-based rules to determine the category"""
    combined_text = str(row['Omschrijving'] if not pd.isna(row['Omschrijving']) else '') + ' ' + \
                    str(row['Omschrijving11'] if not pd.isna(row['Omschrijving11']) else '')
    combined_text = combined_text.lower()
    
    if 'koeler' in combined_text:
        return 'SF-VKS01', 1.0  # 100% confidence for keyword rule
    elif 'verwarming' in combined_text:
        return 'SF-VKS02', 1.0  # 100% confidence for keyword rule
    else:
        return None, None  # No keyword match


def main(input_file=None):
    if input_file is None:
        input_file = notifications_path
    
    # Create default output path in notifications folder
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_dir, 'data', 'notifications')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path with fixed name 'vks_classified.xlsx'
    output_file = os.path.join(output_dir, 'vks_classified.xlsx')
    
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output will be saved to: {output_file}")

    # Load data
    df = load_data(input_file)
    logger.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

    # Get training data for each category
    categories = ['SF-VKS01', 'SF-VKS02', 'SF-VKS03']
    vks_categories = categories + ['SF-VKS']  # All VKS categories including generic
    
    # Filter for records with specific SF-VKS categories for training
    training_data = df[df['Verantw.werkpl.'].isin(categories)].copy()
    logger.info(f"Found {training_data.shape[0]} records with specific SF-VKS categories for training")
    
    # Count distribution of categories in training data
    for category in categories:
        count = (training_data['Verantw.werkpl.'] == category).sum()
        logger.info(f"  - {category}: {count} training records")
    
    # Train separate classifiers for each category
    models = {}
    for category in categories:
        models[category] = train_binary_classifier(training_data.copy(), category)
    
    # Get records to classify (all SF-VKS categories, both generic and specific)
    all_vks_records = df[df['Verantw.werkpl.'].isin(vks_categories)].copy()
    logger.info(f"Processing all {all_vks_records.shape[0]} VKS records for confidence scores")
    
    # Initialize new columns
    df['Classification_Confidence'] = np.nan
    df['Prediction_Details'] = ''
    
    if all_vks_records.shape[0] > 0:
        # Prepare text features
        all_vks_records['combined_text'] = all_vks_records[['Omschrijving', 'Omschrijving11']].apply(
            lambda row: ' '.join([str(val) if not pd.isna(val) else '' for val in row]), 
            axis=1
        )
        
        # First, apply keyword rules to all VKS records
        logger.info("Applying keyword rules...")
        keyword_results = all_vks_records.apply(apply_keyword_rules, axis=1)
        all_vks_records['keyword_category'] = [result[0] for result in keyword_results]
        all_vks_records['keyword_confidence'] = [result[1] for result in keyword_results]
        
        # Count records affected by keyword rules
        keyword_matches = all_vks_records['keyword_category'].notna().sum()
        logger.info(f"Found {keyword_matches} records matching keyword rules")
        for category in categories:
            count = (all_vks_records['keyword_category'] == category).sum()
            logger.info(f"  - {category} keyword matches: {count} records")
        
        # For records not matched by keywords, predict with each model and get probabilities
        for category in categories:
            proba_col = f"{category}_probability"
            all_vks_records[proba_col] = models[category][0].predict_proba(all_vks_records['combined_text'])[:, 1]
        
        # Determine classification based on highest probability for records without keyword match
        all_vks_records['predicted_category'] = all_vks_records[[f"{cat}_probability" for cat in categories]].idxmax(axis=1)
        all_vks_records['predicted_category'] = all_vks_records['predicted_category'].apply(lambda x: x.replace('_probability', ''))
        all_vks_records['predicted_confidence'] = all_vks_records[[f"{cat}_probability" for cat in categories]].max(axis=1)
        
        # Apply confidence threshold of 80% (0.8)
        confidence_threshold = 0.8
        
        # Final decision logic
        def determine_final_category(row):
            # Only modify generic SF-VKS records, keep others unchanged
            if row['Verantw.werkpl.'] != 'SF-VKS':
                return row['Verantw.werkpl.'], 1.0  # Keep original with 100% confidence
            
            # For SF-VKS records, apply keyword rules first
            if pd.notna(row['keyword_category']):
                return row['keyword_category'], row['keyword_confidence']
            
            # For remaining SF-VKS records, use model prediction if confidence is high enough
            if row['predicted_confidence'] >= confidence_threshold:
                return row['predicted_category'], row['predicted_confidence']
            
            # Keep as SF-VKS if confidence is too low
            return 'SF-VKS', row['predicted_confidence']
        
        # Apply the final decision logic
        final_results = all_vks_records.apply(determine_final_category, axis=1)
        all_vks_records['final_category'] = [result[0] for result in final_results]
        all_vks_records['final_confidence'] = [result[1] for result in final_results]
        
        # Store original values to calculate replacements
        original_values = all_vks_records['Verantw.werkpl.'].copy()
        
        # Calculate total replacements (excluding keyword-forced ones that were already correct)
        keyword_correct = ((all_vks_records['Verantw.werkpl.'] == all_vks_records['keyword_category']) & 
                           pd.notna(all_vks_records['keyword_category'])).sum()
        
        total_replacements = (all_vks_records['final_category'] != original_values).sum()
        effective_replacements = total_replacements - keyword_correct
        
        # Prepare prediction details for each record
        def get_prediction_details(row):
            if row['Verantw.werkpl.'] != 'SF-VKS':
                return ''  # No details for non-SF-VKS records
            
            if pd.notna(row['keyword_category']):
                return f"Keyword match: {row['keyword_category']}"
            
            # Get probability for predicted category
            predicted_cat = row['predicted_category']
            prob = row[f"{predicted_cat}_probability"]
            return f"Model prediction: {predicted_cat} ({prob:.1%})"
        
        # Add prediction details
        all_vks_records['prediction_details'] = all_vks_records.apply(get_prediction_details, axis=1)
        
        # Update original dataframe
        df.loc[all_vks_records.index, 'Verantw.werkpl.'] = all_vks_records['final_category']
        df.loc[all_vks_records.index, 'Classification_Confidence'] = all_vks_records['final_confidence']
        df.loc[all_vks_records.index, 'Prediction_Details'] = all_vks_records['prediction_details']
        
        # Log results
        logger.info("Classification Results Summary:")
        logger.info(f"Total VKS records processed: {all_vks_records.shape[0]}")
        logger.info(f"Records matched by keyword rules: {keyword_matches}")
        
        # Generic SF-VKS specific stats
        generic_records = all_vks_records[original_values == 'SF-VKS']
        logger.info(f"Generic SF-VKS records: {generic_records.shape[0]}")
        logger.info(f"Records classified with confidence ≥{confidence_threshold*100}%: {(generic_records['predicted_confidence'] >= confidence_threshold).sum()} records")
        logger.info(f"Total records category changed: {total_replacements}")
        logger.info(f"  - Due to keywords: {keyword_matches}")
        logger.info(f"  - Due to model prediction: {effective_replacements}")
        
        # Report on distribution of final categories and model predictions
        logger.info("\nClassification Performance and Distribution:")
        for category in categories:
            # Get F1 score from cross-validation results
            f1_mean = np.mean(models[category][1]['f1'])
            logger.info(f"{category}: F1 score of {f1_mean:.3f}")

        logger.info("\nFinal Distribution Details:")
        for category in vks_categories:
            count = (all_vks_records['final_category'] == category).sum()
            percentage = (count / all_vks_records.shape[0]) * 100
            original = (original_values == category).sum()
            change = count - original
            
            # Count predictions (excluding records that were already this category)
            predicted = ((all_vks_records['final_category'] == category) & 
                       (original_values != category)).sum()
            
            logger.info(f"{category}:")
            logger.info(f"  - Total records: {count:,} ({percentage:.2f}%)")
            logger.info(f"  - Original records: {original:,}")
            logger.info(f"  - New predictions: {predicted:,}")
            logger.info(f"  - Net change: {change:+,}")
        
        # Confidence distribution
        confidence_ranges = [
            (0.9, 1.01, '90-100%'),  # Using 1.01 to include 1.0
            (0.8, 0.9, '80-90%'),
            (0.7, 0.8, '70-80%'),
            (0.6, 0.7, '60-70%'),
            (0.0, 0.6, '<60%')
        ]
        
        logger.info("Confidence distribution:")
        for low, high, label in confidence_ranges:
            count = ((all_vks_records['final_confidence'] >= low) & (all_vks_records['final_confidence'] < high)).sum()
            percentage = (count / all_vks_records.shape[0]) * 100
            logger.info(f"  - {label}: {count} records ({percentage:.2f}%)")
        
        # Confidence distribution - alleen voor oorspronkelijke SF-VKS records
        generic_vks_records = all_vks_records[original_values == 'SF-VKS']
        total_generic = len(generic_vks_records)
        
        logger.info("\nConfidence distribution for predicted SF-VKS records only:")
        confidence_ranges = [
            (0.9, 1.01, '90-100%'),  # Using 1.01 to include 1.0
            (0.8, 0.9, '80-90%'),
            (0.7, 0.8, '70-80%'),
            (0.6, 0.7, '60-70%'),
            (0.0, 0.6, '<60%')
        ]
        
        for low, high, label in confidence_ranges:
            count = ((generic_vks_records['final_confidence'] >= low) & 
                    (generic_vks_records['final_confidence'] < high)).sum()
            percentage = (count / total_generic) * 100
            logger.info(f"  - {label}: {count} records ({percentage:.2f}% of SF-VKS records)")
            
        # Voeg extra statistieken toe over de voorspellingen
        logger.info("\nPrediction details for SF-VKS records:")
        logger.info(f"Total original SF-VKS records: {total_generic}")
        
        keyword_matches_vks = generic_vks_records['keyword_category'].notna().sum()
        model_predictions = ((generic_vks_records['final_category'] != 'SF-VKS') & 
                           (generic_vks_records['keyword_category'].isna())).sum()
        remained_generic = (generic_vks_records['final_category'] == 'SF-VKS').sum()
        
        logger.info(f"  - Classified by keywords: {keyword_matches_vks} ({keyword_matches_vks/total_generic*100:.2f}%)")
        logger.info(f"  - Classified by model: {model_predictions} ({model_predictions/total_generic*100:.2f}%)")
        logger.info(f"  - Remained as SF-VKS: {remained_generic} ({remained_generic/total_generic*100:.2f}%)")
    
    # Save results
    logger.info(f"Saving updated dataset to {output_file}")
    df.to_excel(output_file, index=False)
    logger.info("Classification task completed successfully!")


if __name__ == "__main__":
    main()  # Remove output_file parameter, let the function handle it

