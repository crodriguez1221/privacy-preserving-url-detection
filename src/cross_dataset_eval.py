# -*- coding: utf-8 -*-
# --------------------------------------------------------------
# src/cross_dataset_eval.py
# --------------------------------------------------------------
# Purpose:
#   Cross-dataset generalisation evaluation.
#
#   This script answers the core research question:
#   "Do models trained on one real-world threat intelligence
#    source generalise to a completely different one?"
#
#   It trains each model on one dataset and tests it on the
#   other — the model never sees any of the test data during
#   training.  This is a much stricter evaluation than the
#   within-dataset split used in train_models.py.
#
#   Two cross-evaluations are performed:
#       1. Train on PhishTank  -> Test on URLhaus
#       2. Train on URLhaus    -> Test on PhishTank
#
#   Models evaluated:
#       1. Logistic Regression
#       2. Random Forest
#       3. Decision Tree
#
#   Outputs saved to outputs/:
#       - cross_eval_results.csv
#           Structured results table for your writeup
#       - dashboard__cross_eval_phishtank_to_urlhaus.png
#           Visual dashboard for direction 1
#       - dashboard__cross_eval_urlhaus_to_phishtank.png
#           Visual dashboard for direction 2
#       - cross_eval_results_YYYYMMDD_HHMMSS.txt
#           Full terminal output saved automatically
#
# Input files (produced by extract_features.py):
#   data/features_phishtank_tranco_dataset.csv
#   data/features_urlhaus_tranco_dataset.csv
#
# Usage:
#   python src/cross_dataset_eval.py
#   python src/cross_dataset_eval.py --testsize 1.0
# --------------------------------------------------------------

import os # for file path handling
import sys # for sys.exit if something goes wrong
import logging # for logging progress and errors
import argparse # for passing options via command-line (argument parsing)
import traceback # for detailed error tracebacks in logs
import datetime # for timestamped log files and output names

import pandas as pd # for data manipulation and saving spreadsheets style data (CSV files)
import numpy as np # for numerical operations and array handling
import matplotlib.pyplot as plt # for plotting the dashboard and visualizations
import matplotlib.gridspec as gridspec # for flexible dashboard layout
import seaborn as sns # for enhanced visualizations and styling of the dashboard

from sklearn.preprocessing import StandardScaler # for feature scaling (important for models like Logistic Regression)
from sklearn.linear_model  import LogisticRegression # for a simple linear model baseline
from sklearn.ensemble      import RandomForestClassifier # for a high-powered ensemble model that can capture complex patterns
from sklearn.tree          import DecisionTreeClassifier # for a simple interpretable model to compare against the ensemble
from sklearn.metrics       import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
) # for evaluating model performance with various metrics and confusion matrices


# --------------------------------------------------------------
# Feature files — produced by extract_features.py
# --------------------------------------------------------------
PHISHTANK_FILE = os.path.join("data", "features_phishtank_tranco_dataset.csv") # features extracted from PhishTank dataset
URLHAUS_FILE   = os.path.join("data", "features_urlhaus_tranco_dataset.csv") # features extracted from URLhaus dataset

OUTPUT_DIR       = "outputs" # directory to save results and dashboards
LABEL_CANDIDATES = ["label", "class", "target", "type", "result", "status"] # possible column names for the label in the datasets, checked in a case-insensitive way
LABEL_MAP        = {
    "phishing": 1, "phish": 1, "bad": 1, "malicious": 1, "spam": 1, "1": 1,
    "legitimate": 0, "benign": 0, "good": 0, "safe": 0, "0": 0,
}                                                                           # mapping of various label values to a binary format (1 for malicious, 0 for legitimate), applied after normalizing to lowercase and stripping whitespace
METRIC_COLOURS = ["#2E75B6", "#2EA043", "#E67E22", "#8E44AD"]   # consistent colors for the metrics in the dashboard (blue for accuracy, green for precision, orange for recall, purple for F1-score)


# --------------------------------------------------------------
# Tee — mirrors all stdout to a file simultaneously
# --------------------------------------------------------------
class _Tee:
    """Captures all print() and log output to a file and terminal."""
    def __init__(self, file_path: str): # Initialize the _Tee object with the file path to save output
        self._terminal = sys.stdout     # Keep reference to original stdout for terminal output
        self._file     = open(file_path, "w", encoding="utf-8")  # Open the file for writing with UTF-8 encoding

    def write(self, message: str):  # Write a message to both the terminal and the file
        self._terminal.write(message) # Write to terminal
        self._file.write(message)  # Write to file

    def flush(self):  # Flush both the terminal and file buffers to ensure all output is written immediately
        self._terminal.flush()  # Flush terminal buffer
        self._file.flush()  # Flush file buffer

    def close(self):  # Close the file when done to free resources
        self._file.close()  # Close the file


def setup_logging(output_dir: str) -> str:  # Set up logging to capture all output to a timestamped file in the outputs/ directory
    """
    Redirect stdout so all print() and log output is saved to a
    timestamped text file in outputs/ as well as the terminal.

    Saved to: outputs/cross_eval_results_YYYYMMDD_HHMMSS.txt
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Create a timestamp for the log file name
    log_path  = os.path.join(output_dir, f"cross_eval_results_{timestamp}.txt")  # Full path for the log file
     
    sys.stdout = _Tee(log_path)  # Redirect stdout to the _Tee object to capture all output to the log file and terminal

    logger = logging.getLogger()  # Get the root logger
    logger.setLevel(logging.INFO)  # Set logging level to INFO to capture all relevant messages
    handler = logging.StreamHandler(sys.stdout)  # Create a logging handler that writes to the redirected stdout (which goes to both terminal and file)
    handler.setFormatter(logging.Formatter(    # Set a consistent log message format with timestamp and log level
        "%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"
    ))
    logger.addHandler(handler)  # Add the handler to the logger to enable logging to both terminal and file

    logging.getLogger(__name__).info(  # Log the location of the results file for easy reference
        "Results will be saved -> %s", os.path.abspath(log_path)
    )
    return log_path  # Return the path to the log file for reference


log = logging.getLogger(__name__)  # Create a logger for this module to use for logging messages throughout the script


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:   # Load a dataset from a CSV file, trying UTF-8 encoding first and falling back to latin-1 if there is a UnicodeDecodeError
    """Load a feature CSV with UTF-8 / latin-1 encoding fallback."""
    try:
        return pd.read_csv(path, low_memory=False) # Try to read the CSV file with default UTF-8 encoding, setting low_memory=False to avoid dtype inference issues with large files
    except UnicodeDecodeError:  # If there is a UnicodeDecodeError, try again with latin-1 encoding which can handle a wider range of byte sequences without error
        return pd.read_csv(  # Read the CSV file again, this time specifying latin-1 encoding and setting on_bad_lines="skip" to skip any lines that still cause issues, and low_memory=False to avoid dtype inference issues
            path, encoding="latin-1",  # Specify latin-1 encoding to handle files that may not be valid UTF-8
            on_bad_lines="skip", low_memory=False  # Skip lines that cause issues and avoid dtype inference problems with large files
        )


def find_label_column(df: pd.DataFrame) -> str | None:  # Find the label column in the DataFrame by checking for common label names in a case-insensitive way, returning the actual column name if found or None if not found
    lower_map = {c.lower(): c for c in df.columns}  # Create a mapping of lowercase column names to their original names for case-insensitive lookup
    for key in LABEL_CANDIDATES:  # Check each candidate label name in lowercase against the mapping to find the actual column name in the DataFrame
        if key in lower_map:  # If the candidate label name is found in the mapping, return the original column name from the DataFrame
            return lower_map[key]  # Return the original column name that matches the candidate label name (case-insensitive)
    return None  # If no label column is found after checking all candidates, return None to indicate that the label column could not be identified


def normalize_labels(series: pd.Series) -> pd.Series:  # Normalize the label series to a binary format (1 for malicious, 0 for legitimate) by checking if it's numeric and converting it, or mapping string labels using the LABEL_MAP after normalizing to lowercase and stripping whitespace
    if pd.api.types.is_numeric_dtype(series):   # If the series is already numeric, convert it to numeric type, coercing any non-numeric values to NaN (which will be filtered out later)
        return pd.to_numeric(series, errors="coerce")  # Convert the series to numeric, coercing errors to NaN (non-numeric values will become NaN, which can be filtered out later)
    return series.astype(str).str.strip().str.lower().map(LABEL_MAP)  # If the series is not numeric, convert it to string, strip whitespace, convert to lowercase, and map using the LABEL_MAP to convert various label formats to a binary format (1 for malicious, 0 for legitimate)


def prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:  # Load a feature CSV from the given path, identify the label column, normalize the labels to binary format, filter out any rows with invalid labels, and return the feature matrix X and label vector y ready for training or testing
    """
    Load a feature CSV and return (X, y) ready for training or testing.
    Uses the full dataset — no train/test split applied here.
    The split is handled by using entire datasets against each other.
    """
    if not os.path.exists(path):  # Check if the specified feature file exists at the given path, and if not, log an error message and exit the program with a non-zero status code to indicate failure
        log.error("Feature file not found: %s", path)  # If the file does not exist, log an error message indicating that the feature file was not found at the specified path
        log.error("Run extract_features.py first.")  # Log an additional error message suggesting to run the extract_features.py script first, which is responsible for generating the feature CSV files needed for this evaluation
        sys.exit(1)  # Exit the program with a non-zero status code to indicate that an error occurred (in this case, the required feature file was not found)

    df        = load_dataset(path)  # Load the dataset from the specified path using the load_dataset helper function, which handles UTF-8 and latin-1 encoding fallbacks to ensure the file can be read successfully
    label_col = find_label_column(df)  # Identify the label column in the loaded DataFrame using the find_label_column helper function, which checks for common label column names in a case-insensitive way and returns the actual column name if found or None if not found

    if label_col is None:  # If no label column is found in the DataFrame, log an error message indicating that the label column could not be identified in the specified file and exit the program with a non-zero status code to indicate failure
        log.error("No label column found in %s", path)  # If the label column could not be identified, log an error message indicating that no label column was found in the specified file
        sys.exit(1)  # Exit the program with a non-zero status code to indicate that an error occurred (in this case, the label column could not be identified in the dataset)

    y    = normalize_labels(df[label_col])  # Normalize the labels in the identified label column to a binary format (1 for malicious, 0 for legitimate) using the normalize_labels helper function, which checks if the column is numeric and converts it, or maps string labels using the LABEL_MAP after normalizing to lowercase and stripping whitespace
    mask = y.isin([0, 1])   # Create a boolean mask to filter out any rows where the normalized labels are not valid (i.e., not 0 or 1), ensuring that only rows with valid binary labels are included in the final dataset used for training and testing
    df   = df[mask].reset_index(drop=True)   # Filter the original DataFrame using the boolean mask to keep only rows with valid labels, and reset the index to ensure it is sequential and clean after filtering
    y    = y[mask].reset_index(drop=True).astype(int)  # Apply the same boolean mask to the label series to keep only valid labels, reset the index, and convert the labels to integer type (0 and 1) for use in model training and evaluation

    feature_cols = [  # Select all columns that are not the label column or "url" (which is not a feature) to be used as features for training and testing the models. This assumes that all other columns are numeric features extracted from the URLs.
        c for c in df.columns  # Create a list of feature columns by including all columns in the DataFrame except for the identified label column and the "url" column, which is not a feature but rather an identifier for the URL. This ensures that only relevant numeric features are included in the feature matrix X used for training and testing the models.
        if c not in (label_col, "url")  # Exclude the label column and "url" column from the list of feature columns, as the label column is the target variable for prediction and the "url" column is not a feature but rather an identifier for the URL. All other columns are assumed to be numeric features extracted from the URLs that can be used for training and testing the models.
    ]
    X = df[feature_cols].select_dtypes(include=["number"])  # Create the feature matrix X by selecting only the columns identified as features and ensuring that only numeric columns are included, as the models being trained (Logistic Regression, Random Forest, Decision Tree) require numeric input features. This will automatically exclude any non-numeric columns that may have been included in the feature columns list, ensuring that X contains only valid numeric features for model training and evaluation.

    log.info(  # Log the number of rows, features, and class distribution in the loaded dataset for reference and to confirm that the dataset has been loaded correctly with valid labels. This provides a quick summary of the dataset being used for training or testing in the cross-evaluation.
        "Loaded %s: %d rows, %d features | "
        "malicious: %d (%.1f%%)  legitimate: %d (%.1f%%)",
        os.path.basename(path), len(X), len(X.columns),  # Log the name of the loaded file, number of rows, and number of features in the dataset
        int(y.sum()), 100 * y.mean(),  # Log the number and percentage of malicious samples in the dataset (where y.sum() gives the count of 1s which represent malicious samples, and y.mean() gives the proportion of 1s which can be multiplied by 100 to get the percentage)
        int((y == 0).sum()), 100 * (1 - y.mean()),  # Log the number and percentage of legitimate samples in the dataset (where (y == 0).sum() gives the count of 0s which represent legitimate samples, and (1 - y.mean()) gives the proportion of 0s which can be multiplied by 100 to get the percentage)
    )
    return X, y  # Return the feature matrix X and label vector y ready for training or testing in the cross-evaluation


# --------------------------------------------------------------
# Cross-evaluation for one train→test direction
# --------------------------------------------------------------
def run_cross_eval(  # Train models on the full training dataset and evaluate on the full test dataset, returning a list of per-model result dicts for dashboard rendering. This function handles one direction of the cross-evaluation (e.g., train on PhishTank and test on URLhaus) and is called twice in the main function for both directions.
    train_name: str, # Name of the training dataset (e.g., "PhishTank") for logging and dashboard titles
    test_name:  str, # Name of the testing dataset (e.g., "URLhaus") for logging and dashboard titles
    X_train: pd.DataFrame, # Feature matrix for the training dataset, containing only numeric features and no label or URL columns, ready for model training
    y_train: pd.Series, # Label vector for the training dataset, containing binary labels (0 for legitimate, 1 for malicious) corresponding to the rows in X_train, ready for model training
    X_test:  pd.DataFrame, # Feature matrix for the testing dataset, containing only numeric features and no label or URL columns, ready for model evaluation. This dataset is completely separate from the training dataset and is used to evaluate the generalization of the models trained on the training dataset.
    y_test:  pd.Series, # Label vector for the testing dataset, containing binary labels (0 for legitimate, 1 for malicious) corresponding to the rows in X_test, ready for model evaluation. This dataset is completely separate from the training dataset and is used to evaluate the generalization of the models trained on the training dataset.
    all_results: list, # A shared list to append results to for both directions of the evaluation, allowing for a combined results table to be saved at the end. Each entry in this list will be a dictionary containing the results for one model in one direction of the evaluation, which can then be converted to a DataFrame and saved as a CSV file for reference and use in the writeup.
) -> list: # Returns a list of per-model result dicts for dashboard rendering, where each dict contains the performance metrics and confusion matrix counts for one model evaluated on the test dataset after being trained on the training dataset. This list will be used to create the dashboard visualizations and can also be combined with results from the other direction of the evaluation for a comprehensive results table.
    """
    Train each model on the full training dataset and evaluate it
    on the full test dataset.  No data from the test set is ever
    seen during training.

    Returns a list of per-model result dicts for dashboard rendering.
    """
    direction = f"Train: {train_name}  →  Test: {test_name}"

    print(f"\n{'='*60}")
    print(f"CROSS-DATASET EVALUATION")
    print(f"  {direction}") # Log the direction of the cross-evaluation (which dataset is used for training and which is used for testing) for clarity in the terminal output and to provide context for the results that will be printed. This helps to understand which models were trained on which dataset and evaluated on which test dataset, especially since two directions of evaluation are performed in this script.
    print(f"  Train rows : {len(X_train)}") # Log the number of rows in the training dataset for reference, indicating how many samples were used to train the models in this direction of the evaluation. This provides context for the results and helps to understand the size of the training data that the models were trained on before being evaluated on the test dataset.
    print(f"  Test rows  : {len(X_test)}") # Log the number of rows in the testing dataset for reference, indicating how many samples were used to evaluate the models in this direction of the evaluation. This provides context for the results and helps to understand the size of the test data that the models were evaluated on after being trained on the training dataset.
    print(f"  Features   : {X_train.shape[1]}") # Log the number of features in the training dataset for reference, indicating how many numeric features were used to train the models in this direction of the evaluation. This provides context for the results and helps to understand the dimensionality of the feature space that the models were trained on before being evaluated on the test dataset.
    print(f"{'='*60}") 

    model_results = []

    def evaluate(name, model, X_tr, X_te): # Helper function to train a model, make predictions, compute performance metrics, and compile results into a dictionary for dashboard rendering. This function is called for each model being evaluated (Logistic Regression, Random Forest, Decision Tree) and handles the training and evaluation process for that model on the given training and testing datasets.
        model.fit(X_tr, y_train) # Train the model on the training dataset (X_tr and y_train), which is the full training dataset for this direction of the evaluation. The model will learn patterns from the training data that it will then be evaluated on using the test dataset (X_te and y_test) to assess its generalization performance.
        y_pred = model.predict(X_te) # Use the trained model to make predictions on the test dataset (X_te), which is completely separate from the training dataset and is used to evaluate how well the model generalizes to new, unseen data. The predicted labels (y_pred) will be compared to the true labels (y_test) to compute performance metrics and confusion matrix counts for this model in this direction of the evaluation.
        cm     = confusion_matrix(y_test, y_pred) # Compute the confusion matrix for the predictions made by the model on the test dataset, comparing the true labels (y_test) to the predicted labels (y_pred). The confusion matrix will provide counts of true negatives (tn), false positives (fp), false negatives (fn), and true positives (tp), which will be included in the results dictionary for this model and used in the dashboard to show the performance of the model in terms of how many samples were correctly or incorrectly classified as malicious or legitimate.

        result = { # Compile the results for this model into a dictionary, including the training and testing dataset names, model name, performance metrics (accuracy, precision, recall, F1-score), and confusion matrix counts (true negatives, false positives, false negatives, true positives). The metrics are rounded to 4 decimal places for readability in the dashboard.
            "train_on":  train_name, # Include the name of the training dataset in the results dictionary for this model, which will be used in the dashboard to indicate which dataset was used for training this model in this direction of the evaluation.
            "test_on":   test_name, # Include the name of the testing dataset in the results dictionary for this model, which will be used in the dashboard to indicate which dataset was used for testing this model in this direction of the evaluation.
            "model":     name, # Include the name of the model in the results dictionary, which will be used in the dashboard to indicate which model's performance is being shown for this entry in this direction of the evaluation.
            "accuracy":  round(accuracy_score(y_test, y_pred),  4), # Compute the accuracy of the model's predictions on the test dataset by comparing the true labels (y_test) to the predicted labels (y_pred) using the accuracy_score function, and round the result to 4 decimal places for readability in the dashboard. Accuracy represents the overall proportion of correct predictions made by the model on the test dataset.
            "precision": round(precision_score( # Compute the precision of the model's predictions on the test dataset by comparing the true labels (y_test) to the predicted labels (y_pred) using the precision_score function, and round the result to 4 decimal places for readability in the dashboard. Precision represents the proportion of positive predictions made by the model that were actually correct (i.e., how many of the samples predicted as malicious were truly malicious). The zero_division=0 argument is used to handle cases where there may be no positive predictions, preventing a division by zero error and returning a precision of 0 in such cases.
                y_test, y_pred, zero_division=0), 4), # Compute the recall of the model's predictions on the test dataset by comparing the true labels (y_test) to the predicted labels (y_pred) using the recall_score function, and round the result to 4 decimal places for readability in the dashboard. Recall represents the proportion of actual positive samples (malicious) that were correctly identified by the model (i.e., how many of the truly malicious samples were predicted as malicious). The zero_division=0 argument is used to handle cases where there may be no actual positive samples, preventing a division by zero error and returning a recall of 0 in such cases.
            "recall":    round(recall_score( # Compute the F1-score of the model's predictions on the test dataset by comparing the true labels (y_test) to the predicted labels (y_pred) using the f1_score function, and round the result to 4 decimal places for readability in the dashboard. F1-score is the harmonic mean of precision and recall, providing a single metric that balances both precision and recall. The zero_division=0 argument is used to handle cases where there may be no positive predictions or no actual positive samples, preventing a division by zero error and returning an F1-score of 0 in such cases.
                y_test, y_pred, zero_division=0), 4), # Compute the F1-score of the model's predictions on the test dataset by comparing the true labels (y_test) to the predicted labels (y_pred) using the f1_score function, and round the result to 4 decimal places for readability in the dashboard. F1-score is the harmonic mean of precision and recall, providing a single metric that balances both precision and recall. The zero_division=0 argument is used to handle cases where there may be no positive predictions or no actual positive samples, preventing a division by zero error and returning an F1-score of 0 in such cases.
            "f1_score":  round(f1_score( # Compute the F1-score of the model's predictions on the test dataset by comparing the true labels (y_test) to the predicted labels (y_pred) using the f1_score function, and round the result to 4 decimal places for readability in the dashboard. F1-score is the harmonic mean of precision and recall, providing a single metric that balances both precision and recall. The zero_division=0 argument is used to handle cases where there may be no positive predictions or no actual positive samples, preventing a division by zero error and returning an F1-score of 0 in such cases.
                y_test, y_pred, zero_division=0), 4), # Compute the F1-score of the model's predictions on the test dataset by comparing the true labels (y_test) to the predicted labels (y_pred) using the f1_score function, and round the result to 4 decimal places for readability in the dashboard. F1-score is the harmonic mean of precision and recall, providing a single metric that balances both precision and recall. The zero_division=0 argument is used to handle cases where there may be no positive predictions or no actual positive samples, preventing a division by zero error and returning an F1-score of 0 in such cases.
            "tn": int(cm[0][0]), "fp": int(cm[0][1]), # Include the counts of true negatives (tn), false positives (fp), false negatives (fn), and true positives (tp) from the confusion matrix in the results dictionary for this model, which will be used in the dashboard to show the performance of the model in terms of how many samples were correctly or incorrectly classified as malicious or legitimate. The counts are converted to integers for consistency and readability in the dashboard.
            "fn": int(cm[1][0]), "tp": int(cm[1][1]), # Include the counts of true negatives (tn), false positives (fp), false negatives (fn), and true positives (tp) from the confusion matrix in the results dictionary for this model, which will be used in the dashboard to show the performance of the model in terms of how many samples were correctly or incorrectly classified as malicious or legitimate. The counts are converted to integers for consistency and readability in the dashboard.
        }
        model_results.append(result) # Append the results dictionary for this model to the model_results list, which will be used to create the dashboard visualizations for this direction of the evaluation. Each entry in this list corresponds to one model's performance metrics and confusion matrix counts when trained on the training dataset and evaluated on the test dataset for this direction of the evaluation.
        all_results.append(result) # Append the results dictionary for this model to the all_results list, which is a shared list that collects results from both directions of the evaluation. This allows for a combined results table to be created at the end of the script that includes results from both training on PhishTank and testing on URLhaus, as well as training on URLhaus and testing on PhishTank, providing a comprehensive overview of the models' performance across both datasets.

        print(f"\n--- {name} ---")
        print(f"  Accuracy  : {result['accuracy']:.4f}")
        print(f"  Precision : {result['precision']:.4f}")
        print(f"  Recall    : {result['recall']:.4f}")
        print(f"  F1-score  : {result['f1_score']:.4f}")
        print(classification_report( # Print a detailed classification report for this model's performance on the test dataset, showing precision, recall, F1-score, and support for each class (legitimate and malicious). The target_names argument is used to label the classes in the report, and zero_division=0 is used to handle cases where there may be no positive predictions or no actual positive samples, preventing a division by zero error and showing a score of 0 in such cases.
            y_test, y_pred, 
            target_names=["Legitimate", "Malicious"],
            zero_division=0
        ))
        return model # Return the trained model after evaluation, which can be used for further analysis or inspection if needed

    # ----------------------------------------------------------
    # Model 1: Logistic Regression
    # Scaled because LR is sensitive to feature magnitudes.
    # Scaler is fit ONLY on training data — never on test data.
    # ----------------------------------------------------------
    scaler  = StandardScaler() # Initialize a StandardScaler for feature scaling, which is important for models like Logistic Regression that are sensitive to the magnitudes of the features. The scaler will be fit only on the training data to avoid data leakage, and then applied to both the training and testing datasets to ensure that the features are on the same scale for model training and evaluation.
    X_tr_sc = scaler.fit_transform(X_train) # Fit the StandardScaler on the training data and transform it to create the scaled training feature matrix X_tr_sc. This ensures that the scaler learns the mean and standard deviation from the training data only, which prevents data leakage and allows for a fair evaluation of the model's generalization to the test dataset.
    X_te_sc = scaler.transform(X_test) # Transform the test feature matrix X_test using the same scaler that was fit on the training data to create the scaled test feature matrix X_te_sc. This ensures that the test features are scaled using the same parameters (mean and standard deviation) as the training features, which is important for a fair evaluation of the model's performance on the test dataset after being trained on the training dataset.

    evaluate( # Evaluate the Logistic Regression model using the evaluate helper function, which trains the model on the scaled training data and evaluates it on the scaled test data, returning the trained model and appending the results to the model_results and all_results lists for dashboard rendering and combined results saving.
        "Logistic Regression",
        LogisticRegression( # Initialize a Logistic Regression model with specified hyperparameters, including maximum iterations set to 1000 to ensure convergence, random state for reproducibility, and class_weight="balanced" to handle any class imbalance in the training data. The Logistic Regression model is a linear model that can serve as a strong baseline for comparison against the more complex Random Forest and Decision Tree models. Setting class_weight="balanced" helps to handle any class imbalance in the training data by giving more weight to the minority class, which can improve the model's performance on the test dataset if there is an imbalance in the classes. The Logistic Regression model is trained on the scaled feature matrices because it is sensitive to feature magnitudes, allowing for a fair evaluation of the model's performance on the test dataset after being trained on the training dataset with proper feature scaling, without introducing any data leakage from scaling.
            max_iter=1000, random_state=42, # Set maximum iterations to 1000 to ensure convergence of the Logistic Regression model, and set random state for reproducibility of results. The Logistic Regression model is a linear model that can serve as a strong baseline for comparison against the more complex Random Forest and Decision Tree models. Setting class_weight="balanced" helps to handle any class imbalance in the training data by giving more weight to the minority class, which can improve the model's performance on the test dataset if there is an imbalance in the classes.
            class_weight="balanced" # Set class_weight="balanced" to handle any class imbalance in the training data, ensuring that the model gives appropriate attention to both classes during training. The Logistic Regression model is trained on the scaled feature matrices because it is sensitive to feature magnitudes, allowing for a fair evaluation of the model's performance on the test dataset after being trained on the training dataset with proper feature scaling, without introducing any data leakage from scaling.
        ),
        X_tr_sc, X_te_sc, # Train and evaluate the Logistic Regression model on the scaled training and testing datasets, as Logistic Regression is sensitive to feature magnitudes. This allows for a fair evaluation of the model's performance on the test dataset after being trained on the training dataset with proper feature scaling, without introducing any data leakage from scaling.
    )

    # ----------------------------------------------------------
    # Model 2: Random Forest
    # ----------------------------------------------------------
    rf = evaluate( # Evaluate the Random Forest model using the evaluate helper function, which trains the model on the original (unscaled) training data and evaluates it on the original (unscaled) test data, returning the trained model and appending the results to the model_results and all_results lists for dashboard rendering and combined results saving. The Random Forest model is not sensitive to feature scaling, so it is trained and evaluated on the original feature matrices without scaling.
        "Random Forest",
        RandomForestClassifier( # Initialize a Random Forest Classifier with specified hyperparameters, including the number of trees (n_estimators=200), random state for reproducibility, using all available CPU cores for training (n_jobs=-1), and class_weight="balanced" to handle any class imbalance in the training data. The Random Forest model is an ensemble of decision trees that can capture complex patterns in the data and is generally robust to feature scaling, so it is trained on the original feature matrices without scaling.
            n_estimators=200, random_state=42, # Set the number of trees in the forest to 200 for a stronger ensemble that can capture more complex patterns in the data, which may improve performance on the test dataset. Set random state for reproducibility of results, n_jobs=-1 to use all available CPU cores for faster training, and class_weight="balanced" to handle any class imbalance in the training data, ensuring that the model gives appropriate attention to both classes during training. The Random Forest model is trained on the original feature matrices without scaling, as it is not sensitive to feature magnitudes, allowing for a fair evaluation of the model's performance on the test dataset after being trained on the training dataset without introducing any data leakage from scaling.
            n_jobs=-1, class_weight="balanced" # Use all available CPU cores for training to speed up the process, and set class_weight="balanced" to handle any class imbalance in the training data, ensuring that the model gives appropriate attention to both classes during training. The Random Forest model is trained on the original feature matrices without scaling, as it is not sensitive to feature magnitudes, allowing for a fair evaluation of the model's performance on the test dataset after being trained on the training dataset without introducing any data leakage from scaling.
        ),
        X_train, X_test, # Train and evaluate the Random Forest model on the original (unscaled) training and testing datasets, as the Random Forest model is not sensitive to feature scaling. This allows for a fair evaluation of the model's performance on the test dataset after being trained on the training dataset without introducing any data leakage from scaling.
    )

    importances = pd.Series( # Extract the feature importances from the trained Random Forest model and create a pandas Series with the feature names as the index and the importance scores as the values, then sort the Series in descending order to identify which features were most important for the Random Forest model's predictions. This information will be used in the dashboard to show the top 10 most important features according to the Random Forest model trained on the training dataset.
        rf.feature_importances_, index=X_train.columns # Create a pandas Series for feature importances using the feature_importances_ attribute of the trained Random Forest model, with the index set to the column names of the training feature matrix X_train to associate each importance score with its corresponding feature name.
    ).sort_values(ascending=False) # Sort the feature importances in descending order so that the most important features are at the top of the Series, making it easier to identify which features had the greatest impact on the Random Forest model's predictions. This sorted Series will be used to display the top 10 feature importances in the dashboard and to print them in the terminal for reference.

    print("\n  Top 10 Feature Importances (Random Forest):")
    for feat, score in importances.head(10).items(): # Print the top 10 feature importances from the Random Forest model in the terminal, showing the feature name and its corresponding importance score rounded to 4 decimal places for readability. This provides insight into which features were most influential in the Random Forest model's predictions on the test dataset after being trained on the training dataset.
        print(f"    {feat:<35s} {score:.4f}") # Print the feature name left-aligned in a 35-character wide field, followed by the importance score rounded to 4 decimal places, for a clear and organized display of the top 10 feature importances from the Random Forest model in the terminal. This information will also be used in the dashboard to visually show the most important features according to the Random Forest model.

    # ----------------------------------------------------------
    # Model 3: Decision Tree
    # ----------------------------------------------------------
    evaluate( # Evaluate the Decision Tree model using the evaluate helper function, which trains the model on the original (unscaled) training data and evaluates it on the original (unscaled) test data, returning the trained model and appending the results to the model_results and all_results lists for dashboard rendering and combined results saving. The Decision Tree model is not sensitive to feature scaling, so it is trained and evaluated on the original feature matrices without scaling. The hyperparameters specified include a maximum depth of 10 to prevent overfitting, a minimum samples split of 20 to ensure that nodes are not split if they have fewer than 20 samples, random state for reproducibility, and class_weight="balanced" to handle any class imbalance in the training data.
        "Decision Tree (depth=10)",
        DecisionTreeClassifier( # Initialize a Decision Tree Classifier with specified hyperparameters, including a maximum depth of 10 to prevent overfitting, a minimum samples split of 20 to ensure that nodes are not split if they have fewer than 20 samples, random state for reproducibility, and class_weight="balanced" to handle any class imbalance in the training data. The Decision Tree model is a simple interpretable model that can capture non-linear patterns in the data and is generally robust to feature scaling, so it is trained on the original feature matrices without scaling.
            max_depth=10, min_samples_split=20, # Set maximum depth to 10 to prevent overfitting and ensure the tree does not grow too deep, which can lead to better generalization on the test dataset. Set minimum samples split to 20 to ensure that nodes are not split if they have fewer than 20 samples, which can also help prevent overfitting and improve generalization.
            random_state=42, class_weight="balanced" # Set random state for reproducibility of results and class_weight="balanced" to handle any class imbalance in the training data, ensuring that the model gives appropriate attention to both classes during training. The Decision Tree model is trained on the original feature matrices without scaling, as it is not sensitive to feature magnitudes, allowing for a fair evaluation of the model's performance on the test dataset after being trained on the training dataset without introducing any data leakage from scaling.
        ),
        X_train, X_test, # Train and evaluate the Decision Tree model on the original (unscaled) training and testing datasets, as the Decision Tree model is not sensitive to feature scaling. This allows for a fair evaluation of the model's performance on the test dataset after being trained on the training dataset without introducing any data leakage from scaling.
    )

    return model_results, importances # Return the list of model results (performance metrics and confusion matrix counts for each model) and the feature importances from the Random Forest model, which will be used to create the dashboard visualizations for this direction of the evaluation and to show the top 10 most important features according to the Random Forest model in the dashboard.


# --------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------
def show_dashboard( # Build and save a cross-evaluation dashboard using matplotlib and seaborn, with a layout that mirrors the train_models.py dashboard for easy side-by-side comparison. The dashboard includes a metrics table, a grouped bar chart comparing the performance metrics of the models, a horizontal bar chart showing the top 10 feature importances from the Random Forest model, and confusion matrices for each model evaluated. The dashboard is saved as a PNG file in the outputs/ directory with a name that indicates the training and testing datasets used in the evaluation.
    train_name:   str, # Name of the training dataset (e.g., "PhishTank") for logging and dashboard titles
    test_name:    str, # Name of the testing dataset (e.g., "URLhaus") for logging and dashboard titles
    model_results: list, # A list of dictionaries containing the results for each model evaluated, including performance metrics and confusion matrix counts, which will be used to populate the metrics table and bar charts in the dashboard.
    importances:  pd.Series, # A pandas Series containing the feature importances from the Random Forest model, sorted in descending order, which will be used to display the top 10 most important features in the dashboard.
    output_dir:   str, # Directory where the dashboard image will be saved, typically "outputs". The dashboard will be saved as a PNG file with a name that indicates the training and testing datasets used in the evaluation (e.g., "dashboard__cross_eval_phishtank_to_urlhaus.png").
) -> None: # This function does not return anything, but it saves the generated dashboard as a PNG file in the specified output directory. The dashboard visually summarizes the results of the cross-evaluation for the given training and testing datasets, allowing for easy comparison of model performance and feature importances.
    """
    Build and save a cross-evaluation dashboard.
    Layout mirrors train_models.py for easy side-by-side comparison.
    """
    sns.set_theme(style="whitegrid", font_scale=0.9) # Set the Seaborn theme for the dashboard with a white grid background and a slightly smaller font scale for better readability in the dashboard visualizations. This will ensure that the plots and tables in the dashboard have a consistent and visually appealing style that matches the theme used in the train_models.py dashboard for easy comparison.

    fig = plt.figure(figsize=(20, 14)) # Create a new figure for the dashboard with a specified size of 20 inches in width and 14 inches in height, providing ample space for the various subplots (metrics table, bar charts, confusion matrices) to be displayed clearly without overcrowding. The larger figure size allows for better readability of the text and visual elements in the dashboard, making it easier to compare the results across different models and datasets.
    fig.patch.set_facecolor("#0D1117") # Set the background color of the entire figure to a dark shade (#0D1117) to create a visually striking contrast with the white and red elements in the plots and tables, enhancing readability and making the dashboard more visually appealing. The dark background also helps to highlight the colors used for the bars in the bar charts and the cells in the confusion matrices, drawing attention to the key results being presented in the dashboard.

    fig.suptitle( # Set the overall title for the dashboard, indicating that it is a cross-dataset evaluation and specifying which datasets were used for training and testing. The title is formatted to replace underscores with spaces and to capitalize each word for better readability. The title is styled with a larger font size, bold weight, and white color to make it stand out against the dark background of the figure, providing clear context for the results being displayed in the dashboard.
        f"Cross-Dataset Evaluation\n"
        f"Trained on: {train_name.replace('_', ' ').title()}   " # Format the training dataset name by replacing underscores with spaces and capitalizing each word, and include it in the title to indicate which dataset was used for training the models in this direction of the evaluation. This provides context for the results being displayed in the dashboard and allows for easy comparison with the other direction of evaluation where a different dataset is used for training.
        f"→   Tested on: {test_name.replace('_', ' ').title()}", # Format the testing dataset name by replacing underscores with spaces and capitalizing each word, and include it in the title to indicate which dataset was used for testing the models in this direction of the evaluation. This provides context for the results being displayed in the dashboard and allows for easy comparison with the other direction of evaluation where a different dataset is used for testing.
        fontsize=14, fontweight="bold", # Set the font size to 14 and the font weight to bold for the title to make it stand out and be easily readable at the top of the dashboard, providing clear context for the results being presented in the visualizations below.
        color="white", y=0.98, # Set the color of the title text to white to create contrast against the dark background. The y parameter is set to 0.98 to position the title slightly below the top edge of the figure, ensuring that it is clearly visible and does not overlap with any other elements in the dashboard.
    )

    gs = gridspec.GridSpec( # Create a GridSpec layout for the subplots in the dashboard, specifying
        3, 4, figure=fig, # Define a grid with 3 rows and 4 columns for the subplots in the dashboard, allowing for a structured layout that can accommodate the metrics table, bar charts, and confusion matrices in an organized manner. The GridSpec will be used to specify the position and span of each subplot within the grid, ensuring that the visual elements are arranged in a clear and visually appealing way.
        hspace=0.55, wspace=0.4, # Set the horizontal and vertical spacing between the subplots to 0.55 and 0.4 respectively, providing enough space between the different visual elements in the dashboard to prevent overcrowding and ensure that each subplot is clearly distinguishable. The hspace parameter controls the vertical spacing between rows of subplots, while the wspace parameter controls the horizontal spacing between columns of subplots.
        left=0.06, right=0.97, # Set the left and right margins of the entire grid to 0.06 and 0.97 respectively, providing some padding on the sides of the dashboard to prevent the visual elements from being too close to the edges of the figure. This helps to create a more balanced and visually appealing layout for the dashboard, ensuring that the content is well-framed within the figure.
        top=0.91, bottom=0.06, # Set the top and bottom margins of the entire grid to 0.91 and 0.06 respectively, providing some padding at the top and bottom of the dashboard to prevent the visual elements from being too close to the edges of the figure. This helps to create a more balanced and visually appealing layout for the dashboard, ensuring that the content is well-framed within the figure and that there is enough space for the title at the top and any additional information at the bottom if needed.
    )

    model_names   = [r["model"] for r in model_results] # Extract the model names from the model_results list to use as labels in the bar charts and confusion matrix titles in the dashboard. This allows for clear identification of which results correspond to which models in the visualizations.
    metrics       = ["accuracy", "precision", "recall", "f1_score"] # Define the list of performance metrics that will be extracted from the model_results dictionaries and used in the grouped bar chart in the dashboard to compare the performance of the different models. These metrics represent key aspects of model performance, including overall accuracy, precision for the positive class (malicious), recall for the positive class, and the F1-score which balances precision and recall.
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"] # Define the list of performance metrics and their corresponding labels for use in the grouped bar chart in the dashboard. The metrics list contains the keys that will be used to extract the metric values from the model_results dictionaries, while the metric_labels list contains the human-readable labels that will be displayed in the legend of the bar chart for clarity.

    # ── Row 0: Metrics table ─────────────────────────────────────
    ax_table = fig.add_subplot(gs[0, :]) # Create a subplot that spans the entire first row of the grid (Row 0, all columns) to display the metrics table. This allows for a wide area to present the performance metrics for all models in a clear and organized manner at the top of the dashboard, providing an immediate summary of the results before diving into the visual comparisons in the bar charts and confusion matrices below.
    ax_table.set_facecolor("#161B22") # Set the background color of the metrics table subplot to a dark shade (#161B22) to create a distinct area for the table that contrasts with the white text and red accents in the table cells, enhancing readability.
    ax_table.axis("off") # Turn off the axis for the metrics table subplot since we will be using a table to display the metrics and do not need any axes or ticks, allowing for a cleaner presentation of the performance metrics in the dashboard.

    table_data = [ # Prepare the data for the metrics table by creating a list of lists, where each inner list corresponds to a row in the table. Each row contains the model name, accuracy, precision, recall, F1-score, and the total number of test rows (calculated as the sum of true negatives, false positives, false negatives, and true positives) for each model evaluated. The metric values are formatted to 4 decimal places for readability in the dashboard.
        [
            r["model"], # Include the model name in the first column of the table for each row, allowing for clear identification of which performance metrics correspond to which model in the dashboard.
            f"{r['accuracy']:.4f}", # Format the accuracy metric to 4 decimal places and include it in the second column of the table for each row.
            f"{r['precision']:.4f}", # Format the precision metric to 4 decimal places and include it in the third column of the table for each row.
            f"{r['recall']:.4f}", # Format the recall metric to 4 decimal places and include it in the fourth column of the table for each row.
            f"{r['f1_score']:.4f}", # Format the F1-score metric to 4 decimal places and include it in the fifth column of the table for each row.
            f"{r['tp']+r['tn']+r['fp']+r['fn']} rows", # Calculate the total number of test rows by summing the true positives, true negatives, false positives, and false negatives from the confusion matrix counts in the model_results dictionary for each model, format it as a string with "rows" appended, and include it in the sixth column of the table for each row. This provides context for the performance metrics by indicating how many samples were evaluated in the test dataset for each model.
        ]
        for r in model_results # Iterate over each dictionary in the model_results list to extract the relevant information (model name, performance metrics, confusion matrix counts) and format it into a list of lists that will be used to populate the metrics table in the dashboard. Each entry in model_results corresponds to one model's evaluation results, and this loop constructs the data structure needed for the table visualization.
    ]
    col_labels = ["Model", "Accuracy", "Precision",
                  "Recall", "F1-Score", "Test Rows"] # Define the column labels for the metrics table, which will be displayed as the header row in the table visualization in the dashboard. These labels correspond to the data that will be presented in each column of the table, providing clear context for the performance metrics and test row counts being displayed for each model.

    tbl = ax_table.table( # Create a table in the ax_table subplot using the table_data and col_labels defined above. 
        cellText=table_data, # Set the cellText parameter to the table_data list of lists, which contains the performance metrics and test row counts for each model, to populate the cells of the table in the dashboard with the relevant information extracted from the model_results.
        colLabels=col_labels, # Set the colLabels parameter to the col_labels list, which contains the headers for each column in the table, to display the appropriate labels at the top of the table in the dashboard, providing context for the performance metrics and test row counts being presented for each model.
        cellLoc="center",  # Set the cellLoc parameter to "center" to center-align the text within each cell of the table, improving readability and creating a more visually appealing presentation of the performance metrics and test row counts for each model in the dashboard.
        loc="center", # Set the loc parameter to "center" to position the entire table at the center of the ax_table subplot, ensuring that it is prominently displayed and easily readable at the top of the dashboard, providing an immediate summary of the results before diving into the visual comparisons in the bar charts and confusion matrices below.
    )
    tbl.auto_set_font_size(False) # Disable automatic font size adjustment for the table to allow for manual control over the font size, ensuring that the text in the table is large enough to be easily readable in the dashboard while still fitting within the cells appropriately.
    tbl.set_fontsize(10) # Set the font size of the table text to 10 for better readability in the dashboard, ensuring that the performance metrics and test row counts are clearly visible without being too small or too large for the table cells.
    tbl.scale(1, 2.2) # Scale the table to adjust the cell sizes, with a horizontal scale of 1 (no change) and a vertical scale of 2.2 to increase the height of the cells, providing more space for the text and improving readability in the dashboard. This scaling helps to create a more visually appealing presentation of the performance metrics and test row counts for each model in the table.

    for j in range(len(col_labels)): # Iterate over the range of the number of columns in the table (based on the length of col_labels) to apply styling to the header row of the table. This loop allows for customization of the appearance of the header cells, making them visually distinct from the rest of the table and enhancing readability.
        tbl[(0, j)].set_facecolor("#C0392B") # Set the background color of the header cells (row 0) to a red shade (#C0392B) to create a visually striking header that contrasts with the dark background and the white text, making the column labels stand out in the dashboard.
        tbl[(0, j)].set_text_props(color="white", fontweight="bold") # Set the text properties of the header cells (row 0) to have white color and bold font weight, enhancing the visibility and emphasis of the column labels in the dashboard, making it easier for viewers to understand what each column represents in terms of performance metrics and test row counts for each model.

    row_colours = ["#1C2128", "#21262D"] # Define a list of two dark shades to be used for alternating row colors in the table, creating a striped effect that improves readability by visually separating the rows. The first color (#1C2128) will be used for odd rows and the second color (#21262D) will be used for even rows, providing a subtle contrast that helps to distinguish between different models' performance metrics in the dashboard.
    for i in range(len(table_data)): # Iterate over the range of the number of rows in the table (based on the length of table_data) to apply alternating background colors to the data rows of the table. This loop allows for customization of the appearance of the data cells, creating a striped effect that enhances readability in the dashboard by visually separating the rows corresponding to different models.
        for j in range(len(col_labels)): # Iterate over the range of the number of columns in the table (based on the length of col_labels) to apply the background color to each cell in the current row. This nested loop ensures that all cells in a given row receive the same background color, creating a consistent and visually appealing striped effect across the entire row in the dashboard.
            tbl[(i+1, j)].set_facecolor(row_colours[i % 2]) # Set the background color of the data cells (starting from row 1) to alternate between the two colors defined in row_colours based on whether the row index i is even or odd. This creates a striped effect in the table that improves readability by visually separating the rows corresponding to different models' performance metrics in the dashboard.
            tbl[(i+1, j)].set_text_props(color="white") # Set the text color of the data cells (starting from row 1) to white to create contrast against the dark background colors, enhancing readability and ensuring that the performance metrics and test row counts for each model are clearly visible in the dashboard.

    ax_table.set_title( # Set the title for the metrics table subplot, indicating that it is a summary of model performance for the specified training and testing datasets. 
        f"Model Performance  |  Trained on {train_name.replace('_',' ').title()}" # Format the training dataset name by replacing underscores with spaces and capitalizing each word, and include it in the title to indicate which dataset was used for training the models in this direction of the evaluation. This provides context for the results being displayed in the table and allows for easy comparison with the other direction of evaluation where a different dataset is used for training.
        f"  →  Tested on {test_name.replace('_',' ').title()}", # Format the testing dataset name by replacing underscores with spaces and capitalizing each word, and include it in the title to indicate which dataset was used for testing the models in this direction of the evaluation. This provides context for the results being displayed in the table and allows for easy comparison with the other direction of evaluation where a different dataset is used for testing.
        color="white", fontsize=10, fontweight="bold", pad=8 # Set the color of the title text to white for contrast against the dark background, set the font size to 10 for readability, set the font weight to bold to make it stand out, and set the padding to 8 to provide some space between the title and the table for a visually appealing layout in the dashboard.
    )

    # ── Row 1 Col 0-1: Grouped bar chart ─────────────────────────
    ax_bar = fig.add_subplot(gs[1, :2]) # Create a subplot that spans the first two columns of the second row of the grid (Row 1, Columns 0-1) to display the grouped bar chart comparing the performance metrics of the different models. This allows for a clear visual comparison of the accuracy, precision, recall, and F1-score for each model evaluated in this direction of the cross-dataset evaluation, providing insights into which models performed better on the test dataset after being trained on the training dataset.
    ax_bar.set_facecolor("#161B22") # Set the background color of the bar chart subplot to a dark shade (#161B22) to create a visually distinct area for the bar chart that contrasts with the colors of the bars and the white text, enhancing readability and making the performance comparisons stand out in the dashboard.

    x      = np.arange(len(model_names)) # Create an array of x positions for the bars in the grouped bar chart based on the number of models evaluated (length of model_names). This array will be used to position the bars for each model along the x-axis in the bar chart, allowing for a clear visual comparison of the performance metrics across different models in the dashboard.
    width  = 0.18 # Define the width of each bar in the grouped bar chart to ensure that the bars for different metrics are visually distinct and do not overlap, allowing for a clear comparison of the accuracy, precision, recall, and F1-score for each model evaluated in this direction of the cross-dataset evaluation. The width is set to 0.18 to provide enough space for multiple bars per model while maintaining a compact and visually appealing layout in the dashboard.
    offset = -(len(metrics) - 1) / 2 * width # Calculate the offset for the bars in the grouped bar chart to center the bars for each model around their corresponding x position. The offset is calculated based on the number of metrics being compared (length of metrics) and the width of each bar, ensuring that the bars for different metrics are evenly spaced around the x position for each model, creating a visually balanced and organized bar chart in the dashboard.

    for idx, (metric, label, colour) in enumerate( # Iterate over the metrics, their corresponding labels, and colors using enumerate to create the grouped bars for each metric in the bar chart. This loop allows for the creation of separate bars for accuracy, precision, recall, and F1-score for each model evaluated, with distinct colors and labels for clarity in the dashboard.
        zip(metrics, metric_labels, METRIC_COLOURS) # Use zip to iterate over the metrics, their corresponding labels, and colors in parallel, allowing for the creation of bars in the grouped bar chart with the appropriate metric values, labels for the legend, and colors for visual distinction in the dashboard.
    ):
        vals = [r[metric] for r in model_results] # Extract the values for the current metric from the model_results list of dictionaries to use as the heights of the bars in the grouped bar chart for this metric. This allows for a visual comparison of the specific performance metric (accuracy, precision, recall, or F1-score) across all models evaluated in this direction of the cross-dataset evaluation, providing insights into which models performed better on the test dataset for that particular metric.
        bars = ax_bar.bar( # Create a bar for the current metric in the grouped bar chart, positioning it based on the x array, the calculated offset, and the index of the metric to ensure that the bars for different metrics are visually distinct and properly spaced for each model. The height of the bars is determined by the vals list, which contains the metric values for each model, and the bars are styled with the specified color and alpha for visual appeal in the dashboard.
            x + offset + idx * width, vals, # Position the bars for this metric based on the x array, the calculated offset to center the bars for each model, and the index of the metric multiplied by the width to space the bars for different metrics apart. The height of the bars is determined by the vals list, which contains the metric values for each model, allowing for a visual comparison of this specific performance metric across all models evaluated in this direction of the cross-dataset evaluation.
            width, label=label, color=colour, alpha=0.88 # Set the width of the bars to the defined width, assign the label for the legend based on the metric_labels, set the color of the bars based on the METRIC_COLOURS list, and set the alpha for transparency to 0.88 for a visually appealing bar chart in the dashboard that allows for clear comparison of the performance metrics across different models.
        )
        for bar, val in zip(bars, vals): # Iterate over the bars created for this metric and their corresponding values using zip to add text labels on top of each bar that display the metric value rounded to 3 decimal places. This loop allows for the addition of value labels on top of each bar in the grouped bar chart, providing precise numerical information about the performance metrics for each model in the dashboard, enhancing the interpretability of the visual comparison.
            ax_bar.text( # Add a text label on top of each bar in the grouped bar chart to display the metric value for that bar, rounded to 3 decimal places for readability. The text is positioned at the center of the bar's x position and slightly above the top of the bar to ensure that it is clearly visible and does not overlap with the bar itself. The text is styled with a smaller font size and white color for contrast against the dark background and the colors of the bars, making it easy to read in the dashboard.
                bar.get_x() + bar.get_width() / 2, # Position the text at the center of the bar's x position by taking the x position of the bar and adding half of the bar's width.
                bar.get_height() + 0.01, # Position the text slightly above the top of the bar by taking the height of the bar and adding a small offset (0.01) to ensure that the text does not overlap with the bar itself and is clearly visible in the dashboard.
                f"{val:.3f}", # Format the metric value to 3 decimal places and convert it to a string to be displayed as the text label on top of the bar in the grouped bar chart, providing precise numerical information about the performance metric for each model in the dashboard.
                ha="center", va="bottom", # Set the horizontal alignment of the text to "center" to align it with the center of the bar, and set the vertical alignment to "bottom" to position it just above the top of the bar, ensuring that the text is clearly visible and does not overlap with the bar itself in the dashboard.
                fontsize=7, color="white", # Set the font size of the text label to 7 for readability without overcrowding the bar, and set the color to white to create contrast against the dark background and the colors of the bars, making the metric values easy to read in the dashboard.
            )

    ax_bar.set_xticks(x) # Set the x-ticks of the bar chart to the x array, which corresponds to the positions of the models on the x-axis. This allows for proper labeling of the bars for each model in the grouped bar chart, ensuring that viewers can easily identify which bars correspond to which models in the dashboard.
    ax_bar.set_xticklabels( # Set the x-tick labels to the model names, replacing spaces with newlines for better readability in the dashboard. This allows for clear identification of which bars correspond to which models in the grouped bar chart, enhancing the interpretability of the performance comparisons across different models in this direction of the cross-dataset evaluation.
        [m.replace(" ", "\n") for m in model_names], # Replace spaces in the model names with newlines to allow for multi-line x-tick labels, improving readability in the dashboard by preventing long model names from overlapping or being cut off on the x-axis of the bar chart.
        color="white", fontsize=9 # Set the color of the x-tick labels to white for contrast against the dark background, and set the font size to 9 for readability in the dashboard, ensuring that the model names are clearly visible and easy to read on the x-axis of the bar chart.
    )
    ax_bar.set_ylim(0, 1.15) # Set the y-axis limits of the bar chart to range from 0 to 1.15 to provide some space above the tallest bars (which can reach up to 1 for metrics like accuracy) for the text labels, ensuring that all bars and their corresponding metric value labels are fully visible within the plot area of the dashboard.
    ax_bar.set_ylabel("Score", color="white") # Set the y-axis label to "Score" to indicate that the values on the y-axis represent performance metric scores, and set the color of the label to white for contrast against the dark background, enhancing readability in the dashboard.
    ax_bar.set_title( # Set the title for the bar chart subplot, indicating that it is a comparison of performance metrics for the specified training and testing datasets. The title is formatted to replace underscores with spaces and to capitalize each word for better readability. The title is styled with a larger font size, bold weight, and white color to make it stand out against the dark background of the subplot, providing clear context for the performance comparisons being presented in the bar chart.
        "Metrics Comparison", color="white", fontweight="bold" # Set the title of the bar chart to "Metrics Comparison" to indicate that this subplot is comparing the performance metrics of the different models, and set the color to white and font weight to bold to make the title stand out against the dark background, enhancing readability in the dashboard.
    )
    ax_bar.tick_params(colors="white") # Set the color of the tick labels and ticks on the y-axis to white for contrast against the dark background, enhancing readability in the dashboard. This ensures that the y-axis values are clearly visible and easy to read, allowing viewers to accurately interpret the performance metric scores for each model in the bar chart.
    ax_bar.spines[:].set_color("#444") # Set the color of the spines (borders) of the bar chart to a dark gray shade (#444) to create a subtle contrast against the dark background, providing a clear boundary for the plot area without being too visually distracting in the dashboard.
    ax_bar.legend( # Add a legend to the bar chart to indicate which colors correspond to which performance metrics, allowing viewers to easily identify the meaning of each bar in the grouped bar chart. The legend is styled with a dark background and white text for contrast against the dark subplot background, enhancing readability in the dashboard.
        loc="upper right", fontsize=8, # Position the legend in the upper right corner of the subplot and set the font size to 8 for readability without overcrowding the plot area in the dashboard.
        facecolor="#21262D", labelcolor="white" # Set the background color of the legend to a dark shade (#21262D) and the color of the text labels in the legend to white for contrast, enhancing readability and ensuring that viewers can easily identify which colors correspond to which performance metrics in the dashboard.
    )

    # ── Row 1 Col 2-3: Feature importance ────────────────────────
    ax_imp = fig.add_subplot(gs[1, 2:]) # Create a subplot that spans the last two columns of the second row of the grid (Row 1, Columns 2-3) to display the horizontal bar chart showing the top 10 feature importances from the Random Forest model. This allows for a clear visual representation of which features were most influential in the Random Forest model's predictions after being trained on the training dataset and tested on the test dataset, providing insights into which features may be most relevant for phishing detection in this cross-dataset evaluation.
    ax_imp.set_facecolor("#161B22") # Set the background color of the feature importance subplot to a dark shade (#161B22) to create a visually distinct area for the bar chart that contrasts with the colors of the bars and the white text, enhancing readability and making the feature importance results stand out in the dashboard.

    top_n = importances.head(10) # Extract the top 10 feature importances from the provided pandas Series, which is assumed to be sorted in descending order. This will allow for a focused visualization of the most influential features in the Random Forest model's predictions, providing insights into which features may be most relevant for phishing detection in this cross-dataset evaluation.
    bars  = ax_imp.barh( # Create a horizontal bar chart to display the top 10 feature importances, with the feature names on the y-axis and their corresponding importance scores on the x-axis. The bars are colored with a red shade and have some transparency for visual appeal in the dashboard. The feature names are displayed in reverse order (from least important to most important) to align with the horizontal bar chart convention where the most important feature is at the top.
        top_n.index[::-1], top_n.values[::-1], # Set the y-axis labels to the feature names (index of the top_n Series) in reverse order to display the most important feature at the top of the horizontal bar chart, and set the x-axis values to the corresponding importance scores (values of the top_n Series) in reverse order to match the feature names. This creates a visually intuitive representation of feature importance in the dashboard, allowing viewers to easily identify which features were most influential in the Random Forest model's predictions.
        color="#C0392B", alpha=0.85 # Set the color of the bars to a red shade (#C0392B) and the alpha for transparency to 0.85 for a visually appealing bar chart in the dashboard that highlights the most important features while maintaining readability against the dark background.
    )
    for bar, val in zip(bars, top_n.values[::-1]): # Iterate over the bars created for the feature importance chart and their corresponding importance values using zip to add text labels on each bar that display the importance score rounded to 4 decimal places. This loop allows for the addition of value labels on each bar in the horizontal bar chart, providing precise numerical information about the importance of each feature in the Random Forest model's predictions, enhancing the interpretability of the feature importance results in the dashboard.
        ax_imp.text( # Add a text label on each bar in the horizontal bar chart to display the importance score for that feature, rounded to 4 decimal places for readability. The text is positioned at the end of the bar (x position) and centered vertically on the bar (y position) to ensure that it is clearly visible and does not overlap with the bar itself. The text is styled with a smaller font size and white color for contrast against the dark background and the red bars, making it easy to read in the dashboard.
            bar.get_width() + 0.001, # Position the text slightly to the right of the end of the bar by taking the width of the bar and adding a small offset (0.001) to ensure that the text does not overlap with the bar itself and is clearly visible in the dashboard.
            bar.get_y() + bar.get_height() / 2, # Position the text vertically centered on the bar by taking the y position of the bar and adding half of the bar's height.
            f"{val:.4f}", va="center", # Set the vertical alignment of the text to "center" to align it with the center of the bar, ensuring that the importance score labels are clearly visible and do not overlap with the bars in the dashboard.
            fontsize=8, color="white" # Set the font size of the text label to 8 for readability without overcrowding the bar, and set the color to white to create contrast against the dark background and the red bars, making the importance scores easy to read in the dashboard.
        )

    ax_imp.set_xlabel("Importance Score", color="white") # Set the x-axis label to "Importance Score" to indicate that the values on the x-axis represent the importance scores of the features, and set the color of the label to white for contrast against the dark background, enhancing readability in the dashboard.
    ax_imp.set_title(  # Set the title for the feature importance subplot, indicating that it is showing the top 10 feature importances from the Random Forest model for the specified training dataset. The title is formatted to replace underscores with spaces and to capitalize each word for better readability. The title is styled with a larger font size, bold weight, and white color to make it stand out against the dark background of the subplot, providing clear context for the feature importance results being presented in the bar chart.
        "Top 10 Feature Importances\n(Random Forest — trained on "
        f"{train_name.replace('_',' ').title()})", # Set the title of the feature importance chart to indicate that it is showing the top 10 feature importances from the Random Forest model, and include the name of the training dataset in the title for context. The title is formatted to replace underscores with spaces and to capitalize each word for better readability, and it is styled with a larger font size, bold weight, and white color to make it stand out against the dark background of the subplot, enhancing readability in the dashboard.
        color="white", fontweight="bold" # Set the color of the title text to white for contrast against the dark background, and set the font weight to bold to make the title stand out, enhancing readability in the dashboard.
    )
    ax_imp.tick_params(colors="white") # Set the color of the tick labels and ticks on the x-axis to white for contrast against the dark background, enhancing readability in the dashboard. This ensures that the x-axis values are clearly visible and easy to read, allowing viewers to accurately interpret the importance scores for each feature in the horizontal bar chart.
    ax_imp.spines[:].set_color("#444") # Set the color of the spines (borders) of the feature importance chart to a dark gray shade (#444) to create a subtle contrast against the dark background, providing a clear boundary for the plot area without being too visually distracting in the dashboard.
    ax_imp.set_facecolor("#161B22") # Set the background color of the feature importance subplot to a dark shade (#161B22) to create a visually distinct area for the bar chart that contrasts with the colors of the bars and the white text, enhancing readability and making the feature importance results stand out in the dashboard.

    # ── Row 2: Confusion matrices ─────────────────────────────────
    for idx, result in enumerate(model_results): # Iterate over the model_results list of dictionaries using enumerate to create a confusion matrix subplot for each model evaluated. This loop allows for the visualization of the confusion matrix for each model in the dashboard, providing insights into the true positives, true negatives, false positives, and false negatives for each model's predictions on the test dataset in this direction of the cross-dataset evaluation.
        ax_cm = fig.add_subplot(gs[2, idx]) # Create a subplot for the confusion matrix of the current model in the third row of the grid (Row 2) and the column corresponding to the index of the model (idx). This allows for a side-by-side comparison of the confusion matrices for each model evaluated in this direction of the cross-dataset evaluation, providing insights into the performance of each model in terms of true positives, true negatives, false positives, and false negatives on the test dataset.
        ax_cm.set_facecolor("#161B22") # Set the background color of the confusion matrix subplot to a dark shade (#161B22) to create a visually distinct area for the confusion matrix that contrasts with the colors of the heatmap and the white text, enhancing readability and making the confusion matrix results stand out in the dashboard.

        cm = np.array([ # Create a 2x2 numpy array to represent the confusion matrix for the current model, using the true negatives (tn), false positives (fp), false negatives (fn), and true positives (tp) from the model_results dictionary. This array will be used to create a heatmap visualization of the confusion matrix in the dashboard, allowing viewers to easily interpret the performance of each model in terms of its predictions on the test dataset.
            [result["tn"], result["fp"]], # The confusion matrix results for true negatives and false positives are placed in the first row of the array, representing the counts of legitimate samples that were correctly classified (tn) and malicious samples that were incorrectly classified as legitimate (fp).
            [result["fn"], result["tp"]], # The confusion matrix results for false negatives and true positives are placed in the second row of the array, representing the counts of legitimate samples that were incorrectly classified as malicious (fn) and malicious samples that were correctly classified (tp).
        ])

        sns.heatmap( # Create a heatmap visualization of the confusion matrix using seaborn's heatmap function, displaying the counts of true negatives, false positives, false negatives, and true positives for the current model. The heatmap is styled with a red color map to visually differentiate between higher and lower counts, and the cell annotations display the exact counts for each category in the confusion matrix. The x-tick labels are set to "Legitimate" and "Malicious" to indicate the predicted classes, while the y-tick labels are set to "Legitimate" and "Malicious" to indicate the actual classes. The axes labels and title are styled with white color for contrast against the dark background, enhancing readability in the dashboard.
            cm, annot=True, fmt="d", # Display the confusion matrix values as integers in the heatmap cells with annotations.
            cmap="Reds", # Use a red color map to visually differentiate between higher and lower counts in the confusion matrix, making it easier to interpret the performance of each model in terms of its predictions on the test dataset in the dashboard.
            xticklabels=["Legitimate", "Malicious"], # Set the x-tick labels to indicate the predicted classes in the confusion matrix, with "Legitimate" representing the negative class and "Malicious" representing the positive class. This provides clear context for interpreting the heatmap in the dashboard.
            yticklabels=["Legitimate", "Malicious"], # Set the y-tick labels to indicate the actual classes in the confusion matrix, with "Legitimate" representing the negative class and "Malicious" representing the positive class. This provides clear context for interpreting the heatmap in the dashboard.
            ax=ax_cm, cbar=False, # Set the ax parameter to the current confusion matrix subplot (ax_cm) to display the heatmap in the correct location in the grid, and set cbar to False to hide the color bar legend for the heatmap, as the exact counts are already displayed in the cell annotations, making it unnecessary in the dashboard.
            linewidths=0.5, # Set the linewidths between the cells in the heatmap to 0.5 to create a clear separation between the different categories in the confusion matrix, enhancing readability and making it easier to interpret the performance of each model in terms of its predictions on the test dataset in the dashboard.
            annot_kws={"size": 11, "weight": "bold"}, # Set the annotation keyword arguments to specify the font size and weight for the cell annotations in the heatmap, making the counts in the confusion matrix more prominent and easier to read in the dashboard.
        )
        ax_cm.set_xlabel("Predicted", color="white", fontsize=9) # Set the x-axis label to "Predicted" to indicate that the values on the x-axis represent the predicted classes in the confusion matrix, and set the color of the label to white for contrast against the dark background, enhancing readability in the dashboard. The font size is set to 9 for readability without overcrowding the plot area.
        ax_cm.set_ylabel("Actual", color="white", fontsize=9) # Set the y-axis label to "Actual" to indicate that the values on the y-axis represent the actual classes in the confusion matrix, and set the color of the label to white for contrast against the dark background, enhancing readability in the dashboard. The font size is set to 9 for readability without overcrowding the plot area.
        ax_cm.set_title( # Set the title for the confusion matrix subplot, indicating which model's confusion matrix is being displayed and the training dataset used for that model. The title is formatted to replace underscores with spaces and to capitalize each word for better readability. The title is styled with a larger font size, bold weight, and white color to make it stand out against the dark background of the subplot, providing clear context for the confusion matrix results being presented in the dashboard.
            result["model"].replace(" (depth=10)", "\n(depth=10)"), # Format the model name in the title by replacing " (depth=10)" with a newline followed by "(depth=10)" to improve readability in the dashboard, especially for the Random Forest model which includes the depth information in its name. This formatting allows for a clearer presentation of the model name and its parameters in the title of the confusion matrix subplot.
            color="white", fontweight="bold", fontsize=9 # Set the color of the title text to white for contrast against the dark background, set the font weight to bold to make the title stand out, and set the font size to 9 for readability without overcrowding the plot area in the dashboard.
        )
        ax_cm.tick_params(colors="white", labelsize=8) # Set the color of the tick labels and ticks on both axes to white for contrast against the dark background, and set the font size of the tick labels to 8 for readability without overcrowding the plot area in the dashboard. This ensures that the axis labels for "Legitimate" and "Malicious" are clearly visible and easy to read, allowing viewers to accurately interpret the confusion matrix for each model in this direction of the cross-dataset evaluation.
        for spine in ax_cm.spines.values(): # Iterate over the spines (borders) of the confusion matrix subplot to set their color to a dark gray shade (#444), creating a subtle contrast against the dark background and providing a clear boundary for the plot area without being too visually distracting in the dashboard.
            spine.set_color("#444") # Set the color of the spines (borders) of the confusion matrix subplot to a dark gray shade (#444) to create a subtle contrast against the dark background, providing a clear boundary for the plot area without being too visually distracting in the dashboard.

    # ── Save and show ─────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True) # Ensure that the output directory exists by creating it if it does not already exist, allowing for the saving of the dashboard image without encountering errors due to a missing directory.
    save_name = f"dashboard__cross_eval_{train_name}_to_{test_name}.png" # Define the name of the file to save the dashboard image, incorporating the training and testing dataset names for clarity. The filename is formatted to indicate that it is a dashboard for the cross-evaluation results from training on one dataset and testing on another, making it easy to identify the contents of the file when saved in the output directory.
    save_path = os.path.join(output_dir, save_name) # Create the full path for saving the dashboard image by joining the output directory and the defined save name, ensuring that the file is saved in the correct location with a descriptive name that indicates its contents in the dashboard.
    plt.savefig( # Save the dashboard figure to the specified path with a resolution of 150 DPI, using tight bounding box to minimize whitespace around the figure, and setting the face color of the saved image to match the face color of the figure for consistency in appearance when viewed outside of the dashboard. This allows for a high-quality image of the dashboard to be saved for later reference or sharing, while maintaining the visual style of the dashboard as seen in the application.
        save_path, dpi=150, # Set the resolution of the saved image to 150 dots per inch (DPI) for a high-quality output that is suitable for viewing and sharing, ensuring that the details of the dashboard are preserved in the saved image.
        bbox_inches="tight", # Use tight bounding box to minimize the amount of whitespace around the figure in the saved image, creating a more compact and visually appealing output that focuses on the content of the dashboard without unnecessary margins.
        facecolor=fig.get_facecolor() # Set the face color of the saved image to match the face color of the figure, ensuring that the visual style of the dashboard is preserved in the saved image when viewed outside of the application, maintaining consistency in appearance and enhancing the overall presentation of the results in the dashboard.
    )
    log.info("Dashboard saved -> %s", os.path.abspath(save_path)) # Log the absolute path of the saved dashboard image to provide feedback to the user about where the file has been saved, allowing for easy access to the results of the cross-dataset evaluation in a visual format.
    plt.show() # Display the dashboard figure to the user, allowing them to visually analyze the performance metrics, feature importances, and confusion matrices for the models evaluated in this direction of the cross-dataset evaluation. This provides an interactive way for users to explore the results and gain insights into the performance of the models when trained on one dataset and tested on another in the context of phishing detection.
     

# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main() -> None: # Define the main function that orchestrates the cross-dataset evaluation process, including loading datasets, running evaluations in both directions, generating dashboards, and saving results. This function serves as the entry point for the application and coordinates the various components to achieve the overall goal of evaluating phishing detection models across different datasets.
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure that the output directory exists by creating it if it does not already exist, allowing for the saving of logs, results, and dashboard images without encountering errors due to a missing directory.
    log_path    = setup_logging(OUTPUT_DIR) # Set up logging for the application by calling the setup_logging function with the output directory, which configures the logging to write to a file in the output directory and returns the path to the log file. This allows for detailed logging of the application's execution, including information about dataset loading, evaluation results, and any errors that may occur, providing a valuable resource for debugging and analysis of the cross-dataset evaluation process.
    all_results = [] # Initialize an empty list to store the results of the cross-dataset evaluations for both directions (training on PhishTank and testing on URLhaus, and vice versa). This list will be populated with dictionaries containing the performance metrics and other relevant information for each model evaluated in both directions, allowing for a comprehensive comparison of the results across different training and testing dataset combinations in the final results table and dashboards.

    # Load both full datasets
    log.info("Loading datasets...") # Log the start of the dataset loading process to provide feedback to the user about the progress of the application, indicating that the datasets are being loaded into memory for the cross-dataset evaluation.
    X_phish, y_phish = prepare(PHISHTANK_FILE) # Call the prepare function with the path to the PhishTank dataset file to load and preprocess the data, returning the feature matrix (X_phish) and target vector (y_phish) for the PhishTank dataset. This prepares the data for use in training and testing the models in the cross-dataset evaluation process.
    X_url,   y_url   = prepare(URLHAUS_FILE) # Call the prepare function with the path to the URLhaus dataset file to load and preprocess the data, returning the feature matrix (X_url) and target vector (y_url) for the URLhaus dataset. This prepares the data for use in training and testing the models in the cross-dataset evaluation process, allowing for a comprehensive evaluation of model performance across both datasets.

    # ----------------------------------------------------------
    # Direction 1: Train on PhishTank → Test on URLhaus
    # ----------------------------------------------------------
    results_1, imp_1 = run_cross_eval( # Call the run_cross_eval function to perform the cross-dataset evaluation in the first direction, where the models are trained on the PhishTank dataset and tested on the URLhaus dataset. The function is provided with the training and testing dataset names, feature matrices, target vectors, and the all_results list to store the results. The function returns the results for this direction of evaluation (results_1) and the feature importances from the Random Forest model (imp_1), which will be used for generating the dashboard for this evaluation direction.
        train_name="PhishTank", # Set the name of the training dataset to "PhishTank" for context in the results and dashboard, indicating that the models were trained on the PhishTank dataset in this direction of the cross-dataset evaluation.
        test_name="URLhaus", # Set the name of the testing dataset to "URLhaus" for context in the results and dashboard, indicating that the models were tested on the URLhaus dataset in this direction of the cross-dataset evaluation.
        X_train=X_phish, y_train=y_phish, # Provide the feature matrix (X_phish) and target vector (y_phish) for the PhishTank dataset as the training data for the models in this direction of the cross-dataset evaluation.
        X_test=X_url,   y_test=y_url, # Provide the feature matrix (X_url) and target vector (y_url) for the URLhaus dataset as the testing data for the models in this direction of the cross-dataset evaluation.
        all_results=all_results, # Pass the all_results list to the run_cross_eval function to allow it to append the results of this evaluation direction, enabling a comprehensive collection of results across both directions of the cross-dataset evaluation for later analysis and comparison in the final results table and dashboards.
    )
    show_dashboard( # Call the show_dashboard function to generate and display the dashboard for the first direction of the cross-dataset evaluation, where the models were trained on the PhishTank dataset and tested on the URLhaus dataset. The function is provided with the training and testing dataset names, the results from this evaluation direction (results_1), the feature importances from the Random Forest model (imp_1), and the output directory for saving the dashboard image. This allows for a visual representation of the performance metrics, feature importances, and confusion matrices for each model evaluated in this direction of the cross-dataset evaluation, providing insights into how well the models trained on PhishTank generalize to URLhaus.
        train_name="PhishTank", # Set the name of the training dataset to "PhishTank" for context in the dashboard, indicating that the models were trained on the PhishTank dataset in this direction of the cross-dataset evaluation.
        test_name="URLhaus", # Set the name of the testing dataset to "URLhaus" for context in the dashboard, indicating that the models were tested on the URLhaus dataset in this direction of the cross-dataset evaluation.
        model_results=results_1, # Provide the results from this evaluation direction (results_1) to the show_dashboard function, which contains the performance metrics and confusion matrix information for each model evaluated when trained on PhishTank and tested on URLhaus. This allows the dashboard to display the relevant performance information for each model in this direction of the cross-dataset evaluation.
        importances=imp_1, # Provide the feature importances from the Random Forest model (imp_1) to the show_dashboard function, which will be used to generate the feature importance bar chart in the dashboard for this direction of the cross-dataset evaluation. This allows viewers to gain insights into which features were most influential in the Random Forest model's predictions when trained on PhishTank and tested on URLhaus, enhancing the interpretability of the results in the dashboard.
        output_dir=OUTPUT_DIR,# Provide the feature importances from the Random Forest model (imp_1) to the show_dashboard function, which will be used to generate the feature importance bar chart in the dashboard for this direction of the cross-dataset evaluation. This allows viewers to gain insights into which features were most influential in the Random Forest model's predictions when trained on PhishTank and tested on URLhaus, enhancing the interpretability of the results in the dashboard.
    )

    # ----------------------------------------------------------
    # Direction 2: Train on URLhaus → Test on PhishTank
    # ----------------------------------------------------------
    results_2, imp_2 = run_cross_eval( # Call the run_cross_eval function to perform the cross-dataset evaluation in the second direction, where the models are trained on the URLhaus dataset and tested on the PhishTank dataset. The function is provided with the training and testing dataset names, feature matrices, target vectors, and the all_results list to store the results. The function returns the results for this direction of evaluation (results_2) and the feature importances from the Random Forest model (imp_2), which will be used for generating the dashboard for this evaluation direction.
        train_name="URLhaus",
        test_name="PhishTank",
        X_train=X_url,   y_train=y_url, # Provide the feature matrix (X_url) and target vector (y_url) for the URLhaus dataset as the training data for the models in this direction of the cross-dataset evaluation.
        X_test=X_phish, y_test=y_phish, # Provide the feature matrix (X_phish) and target vector (y_phish) for the PhishTank dataset as the testing data for the models in this direction of the cross-dataset evaluation.
        all_results=all_results, # Pass the all_results list to the run_cross_eval function to allow it to append the results of this evaluation direction, enabling a comprehensive collection of results across both directions of the cross-dataset evaluation for later analysis and comparison in the final results table and dashboards.
    )
    show_dashboard( # Call the show_dashboard function to generate and display the dashboard for the second direction of the cross-dataset evaluation, where the models were trained on the URLhaus dataset and tested on the PhishTank dataset. The function is provided with the training and testing dataset names, the results from this evaluation direction (results_2), the feature importances from the Random Forest model (imp_2), and the output directory for saving the dashboard image. This allows for a visual representation of the performance metrics, feature importances, and confusion matrices for each model evaluated in this direction of the cross-dataset evaluation, providing insights into how well the models trained on URLhaus generalize to PhishTank.
        train_name="URLhaus",
        test_name="PhishTank",
        model_results=results_2, # Provide the results from this evaluation direction (results_2) to the show_dashboard function, which contains the performance metrics and confusion matrix information for each model evaluated when trained on URLhaus and tested on PhishTank. This allows the dashboard to display the relevant performance information for each model in this direction of the cross-dataset evaluation.
        importances=imp_2, # Provide the feature importances from the Random Forest model (imp_2) to the show_dashboard function, which will be used to generate the feature importance bar chart in the dashboard for this direction of the cross-dataset evaluation. This allows viewers to gain insights into which features were most influential in the Random Forest model's predictions when trained on URLhaus and tested on PhishTank, enhancing the interpretability of the results in the dashboard.
        output_dir=OUTPUT_DIR, # Provide the feature importances from the Random Forest model (imp_2) to the show_dashboard function, which will be used to generate the feature importance bar chart in the dashboard for this direction of the cross-dataset evaluation. This allows viewers to gain insights into which features were most influential in the Random Forest model's predictions when trained on URLhaus and tested on PhishTank, enhancing the interpretability of the results in the dashboard.
    )

    # ----------------------------------------------------------
    # Save results table
    # ----------------------------------------------------------
    results_df   = pd.DataFrame(all_results) # Create a pandas DataFrame from the all_results list of dictionaries, which contains the performance metrics and other relevant information for each model evaluated in both directions of the cross-dataset evaluation. This DataFrame will be used to save the results to a CSV file and to display a summary of the results in the console, allowing for easy analysis and comparison of the performance of the models across different training and testing dataset combinations.
    results_path = os.path.join(OUTPUT_DIR, "cross_eval_results.csv") # Define the path for saving the results CSV file by joining the output directory and the filename "cross_eval_results.csv", ensuring that the results are saved in a structured format that can be easily accessed and analyzed later.
    results_df.to_csv(results_path, index=False) # Save the results DataFrame to a CSV file at the specified path without including the index, allowing for a clean and organized presentation of the results that can be easily opened and analyzed in spreadsheet software or other data analysis tools. This provides a permanent record of the cross-dataset evaluation results for future reference and analysis.
    log.info("Cross-eval results saved -> %s", os.path.abspath(results_path)) # Log the absolute path of the saved results CSV file to provide feedback to the user about where the results have been saved, allowing for easy access to the performance metrics and other relevant information for each model evaluated in both directions of the cross-dataset evaluation for later analysis and comparison.

    print(f"\n{'='*60}")
    print("CROSS-DATASET EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(results_df.to_string(index=False)) # Print a summary of the cross-dataset evaluation results in a tabular format to the console, using the to_string method of the DataFrame to display the results without the index. This provides a clear and organized presentation of the performance metrics and other relevant information for each model evaluated in both directions of the cross-dataset evaluation, allowing for easy analysis and comparison of the results directly in the console without needing to open the saved CSV file.

    print("\nCross-dataset evaluation complete.")

    # Close tee and restore stdout
    if isinstance(sys.stdout, _Tee): # Check if sys.stdout is an instance of the _Tee class, which indicates that it is currently being used to write to both the console and a log file. If this condition is true, it means that the logging mechanism is still active and needs to be properly closed to ensure that all log messages are flushed to the file and that the original sys.stdout is restored for normal console output.
        sys.stdout.close() # If sys.stdout is an instance of _Tee, call the close method to close the _Tee object, which will flush any remaining log messages to the log file and release any resources associated with the logging mechanism. This is an important step to ensure that all log messages are properly saved to the file and that there are no issues with file handles or memory leaks related to the logging mechanism.
        sys.stdout = sys.__stdout__ # Restore the original sys.stdout to sys.__stdout__, which is the default standard output stream for the console. This ensures that any subsequent print statements or console output will be directed to the console as normal, allowing for regular interaction with the user and proper display of any messages or results after the logging mechanism has been closed.
    print(f"\nFull results saved -> {os.path.abspath(log_path)}") # Print the absolute path of the saved log file to the console, providing feedback to the user about where the full results of the cross-dataset evaluation have been logged. This allows for easy access to the detailed logs of the application's execution, including information about dataset loading, evaluation results, and any errors that may have occurred during the process, enabling further analysis and debugging if needed.


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------
if __name__ == "__main__": # Check if the script is being run as the main program, which allows for the execution of the main function that orchestrates the cross-dataset evaluation process. This is a common Python idiom that ensures that the code within this block will only be executed when the script is run directly, and not when it is imported as a module in another script.
    try:
        main() # Call the main function to start the cross-dataset evaluation process, which includes loading datasets, running evaluations in both directions, generating dashboards, and saving results. This serves as the entry point for the application and coordinates the various components to achieve the overall goal of evaluating phishing detection models across different datasets.
    except Exception: # Catch any exceptions that occur during the execution of the main function to handle errors gracefully and provide feedback to the user about what went wrong. This allows for better error handling and debugging, ensuring that any issues encountered during the cross-dataset evaluation process are logged and communicated effectively.
        log.error("Unhandled error:\n%s", traceback.format_exc())
        sys.exit(1) # Exit the program with a non-zero status code to indicate that an error occurred during execution, allowing for proper error handling and signaling to any calling processes or scripts that the execution was not successful. This is important for maintaining good practices in error handling and ensuring that issues are properly logged and addressed.
