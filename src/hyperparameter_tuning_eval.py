# -*- coding: utf-8 -*-
# --------------------------------------------------------------
# src/hyperparameter_tuning_eval.py
# --------------------------------------------------------------
# Purpose:
#   Iteration 3 — Grid Search Hyperparameter Tuning and Re-evaluation.
#
#   This script answers the question:
#   "Does systematic hyperparameter optimization of the best
#    feature configuration from Iteration 2 produce measurable
#    improvements over the baseline and feature-selected models?"
#
#   Strategy:
#       1. Load the reduced 13-feature datasets identified in
#          Iteration 2 (threshold=0.01 mean importance).
#       2. Run GridSearchCV with 5-fold stratified cross-validation
#          on each model independently, optimizing for F1-score.
#       3. Record the best hyperparameter configuration for each
#          model on each dataset.
#       4. Retrain all three models using their best parameters
#          and evaluate on within-dataset and cross-dataset
#          directions using the same methodology as Iterations 1-2.
#       5. Produce a three-way comparison table:
#          Iteration 1 (full 24 features, baseline hyperparameters)
#          vs Iteration 2 (reduced 13 features, baseline hyperparameters)
#          vs Iteration 3 (reduced 13 features, tuned hyperparameters)
#       6. Save all results, dashboards, and logs.
#
#   Parameter grids searched:
#       Logistic Regression : C (regularization strength)
#       Random Forest       : n_estimators, max_depth
#       Decision Tree       : max_depth, min_samples_split
#
#   Outputs saved to outputs/:
#       - hyperparameter_tuning_results.csv
#           Full results table for all three iterations
#       - best_params.csv
#           Best hyperparameter configurations per model per dataset
#       - dashboard__hyperparameter_tuning_within.png
#           Within-dataset three-way comparison dashboard
#       - dashboard__hyperparameter_tuning_cross.png
#           Cross-dataset three-way comparison dashboard
#       - hyperparameter_tuning_eval_YYYYMMDD_HHMMSS.txt
#           Full terminal output
#
# Input files (produced by extract_features.py):
#   data/features_phishtank_tranco_dataset.csv
#   data/features_urlhaus_tranco_dataset.csv
#
# Iteration 2 reduced feature set (threshold=0.01):
#   path_length, num_slashes, url_length, num_dots, subdomain_depth,
#   has_https, num_digits, url_entropy, digit_ratio, hostname_length,
#   special_char_ratio, hostname_entropy, has_ip
#
# Usage:
#   python src/hyperparameter_tuning_eval.py
#   python src/hyperparameter_tuning_eval.py --testsize 0.2
# --------------------------------------------------------------

import os        # provides functions for interacting with the operating system, used for file path construction and directory creation
import sys       # provides access to system-specific parameters and functions, used to redirect stdout for logging and to exit on errors
import logging   # provides a flexible framework for emitting log messages with timestamps and severity levels
import argparse  # provides utilities for parsing command-line arguments, used to allow --testsize flag at runtime
import traceback # provides utilities for extracting and formatting stack traces, used to log detailed error information on unhandled exceptions
import datetime  # provides classes for manipulating dates and times, used to generate timestamped log file names

import pandas as pd   # provides high-performance data structures and analysis tools, used for loading CSVs, building DataFrames, and saving results
import numpy as np    # provides support for large multi-dimensional arrays and mathematical functions, used for numerical operations and chart positioning
import matplotlib.pyplot as plt         # provides a MATLAB-like plotting interface, used to create and save dashboard figures
import matplotlib.gridspec as gridspec  # provides a flexible grid layout manager for subplots, used to arrange dashboard panels in a structured multi-row layout
import seaborn as sns                   # provides statistical data visualization built on matplotlib, used to set the dashboard theme

from sklearn.preprocessing   import StandardScaler         # imports the StandardScaler class for feature normalization, used to scale features for Logistic Regression only
from sklearn.linear_model    import LogisticRegression     # imports the Logistic Regression classifier, used as the linear baseline model
from sklearn.ensemble        import RandomForestClassifier  # imports the Random Forest classifier, used as the nonlinear ensemble model
from sklearn.tree            import DecisionTreeClassifier  # imports the Decision Tree classifier, used as the interpretable rule-based model
from sklearn.model_selection import (
    train_test_split,    # imports the function for splitting data into training and testing sets with stratification support
    GridSearchCV,        # imports the exhaustive grid search cross-validation class used to find the optimal hyperparameter combination for each model
    StratifiedKFold,     # imports the stratified k-fold cross-validation splitter that preserves class distribution across all folds
)
from sklearn.metrics import (
    accuracy_score,      # computes the proportion of correct predictions out of all predictions made
    precision_score,     # computes the proportion of predicted positives that are truly positive
    recall_score,        # computes the proportion of actual positives that were correctly identified
    f1_score,            # computes the harmonic mean of precision and recall as a single balanced metric
    confusion_matrix,    # computes the confusion matrix showing true negatives, false positives, false negatives, and true positives
    classification_report, # generates a text report showing per-class precision, recall, F1-score, and support
)

# --------------------------------------------------------------
# Paths
# --------------------------------------------------------------
PHISHTANK_FILE = os.path.join("data", "features_phishtank_tranco_dataset.csv")  # constructs the file path to the PhishTank feature dataset using os.path.join for cross-platform compatibility
URLHAUS_FILE   = os.path.join("data", "features_urlhaus_tranco_dataset.csv")    # constructs the file path to the URLhaus feature dataset using os.path.join for cross-platform compatibility
OUTPUT_DIR     = "outputs"  # defines the output directory name where all results, dashboards, and logs will be saved

# --------------------------------------------------------------
# Reduced feature set from Iteration 2 (threshold=0.01)
# These 13 features survived the mean importance threshold cut
# when averaged across both PhishTank and URLhaus datasets.
# --------------------------------------------------------------
REDUCED_FEATURES = [
    "path_length",       # mean importance 0.2666 across both datasets, strongest and most consistent predictor
    "num_slashes",       # mean importance 0.2260, second most consistent predictor across both threat sources
    "url_length",        # mean importance 0.1041, captures overall URL complexity
    "num_dots",          # mean importance 0.0765, captures subdomain and path dot structure
    "subdomain_depth",   # mean importance 0.0714, measures levels of subdomain nesting above registered domain
    "has_https",         # mean importance 0.0465, binary HTTPS indicator, more important for URLhaus than PhishTank
    "num_digits",        # mean importance 0.0455, count of numeric characters in the full URL string
    "url_entropy",       # mean importance 0.0434, Shannon entropy of the full URL, stronger for PhishTank
    "digit_ratio",       # mean importance 0.0356, proportion of digit characters relative to total URL length
    "hostname_length",   # mean importance 0.0256, length of the hostname component
    "special_char_ratio",# mean importance 0.0187, proportion of special characters relative to total URL length
    "hostname_entropy",  # mean importance 0.0147, Shannon entropy of the hostname component only
    "has_ip",            # mean importance 0.0108, binary indicator of raw IPv4 address as hostname, stronger for URLhaus
]

# --------------------------------------------------------------
# Hyperparameter grids for grid search
# Each grid defines the parameter combinations to exhaustively
# search using 5-fold stratified cross-validation scored by F1.
# --------------------------------------------------------------
PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],  # C is the inverse regularization strength; smaller values apply stronger regularization which can improve generalization, larger values reduce regularization and fit training data more closely
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],       # number of decision trees in the forest; more trees generally improve stability but increase training time
        "max_depth":    [None, 10, 20, 30],    # maximum depth of each tree; None allows trees to grow until all leaves are pure, while integer values constrain depth to prevent overfitting
    },
    "Decision Tree": {
        "max_depth":         [5, 10, 15, 20],  # maximum depth of the decision tree; lower values reduce overfitting and may improve generalization to unseen data
        "min_samples_split": [10, 20, 30, 50], # minimum number of samples required to split an internal node; higher values prevent the tree from learning overly specific patterns
    },
}

LABEL_CANDIDATES = ["label", "class", "target", "type", "result", "status"]  # defines a list of candidate column names to search for when identifying the label column, checked case-insensitively
LABEL_MAP = {
    "phishing": 1, "phish": 1, "bad": 1, "malicious": 1, "spam": 1, "1": 1,  # maps various string representations of the malicious class to integer 1
    "legitimate": 0, "benign": 0, "good": 0, "safe": 0, "0": 0,              # maps various string representations of the legitimate class to integer 0
}
METRIC_COLOURS = ["#2E75B6", "#2EA043", "#E67E22"]  # defines three hex color codes for Iteration 1, Iteration 2, and Iteration 3 bars in comparison charts

# --------------------------------------------------------------
# Baseline results from Iteration 1 (full 24 features, baseline
# hyperparameters) for three-way comparison in Chapter 4.
# These values are taken directly from training_results logs.
# --------------------------------------------------------------
ITER1_WITHIN = {
    "features_phishtank_tranco_dataset": [
        {"model": "Logistic Regression",       "accuracy": 0.9972, "precision": 0.9990, "recall": 0.9955, "f1_score": 0.9972, "n_features": 24, "iteration": "Iteration 1 (Full 24)"},
        {"model": "Random Forest",             "accuracy": 0.9975, "precision": 0.9980, "recall": 0.9970, "f1_score": 0.9975, "n_features": 24, "iteration": "Iteration 1 (Full 24)"},
        {"model": "Decision Tree (depth=10)",  "accuracy": 0.9980, "precision": 0.9985, "recall": 0.9975, "f1_score": 0.9980, "n_features": 24, "iteration": "Iteration 1 (Full 24)"},
    ],
    "features_urlhaus_tranco_dataset": [
        {"model": "Logistic Regression",       "accuracy": 0.9998, "precision": 1.0000, "recall": 0.9995, "f1_score": 0.9997, "n_features": 24, "iteration": "Iteration 1 (Full 24)"},
        {"model": "Random Forest",             "accuracy": 0.9998, "precision": 1.0000, "recall": 0.9995, "f1_score": 0.9997, "n_features": 24, "iteration": "Iteration 1 (Full 24)"},
        {"model": "Decision Tree (depth=10)",  "accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "n_features": 24, "iteration": "Iteration 1 (Full 24)"},
    ],
}

ITER1_CROSS = {
    ("PhishTank", "URLhaus"): [
        {"model": "Logistic Regression",       "accuracy": 0.9991, "precision": 0.9989, "recall": 0.9993, "f1_score": 0.9991, "n_features": 24, "iteration": "Iteration 1 (Full 24)"},
        {"model": "Random Forest",             "accuracy": 1.0000, "precision": 0.9999, "recall": 1.0000, "f1_score": 1.0000, "n_features": 24, "iteration": "Iteration 1 (Full 24)"},
        {"model": "Decision Tree (depth=10)",  "accuracy": 0.9996, "precision": 0.9991, "recall": 1.0000, "f1_score": 0.9996, "n_features": 24, "iteration": "Iteration 1 (Full 24)"},
    ],
    ("URLhaus", "PhishTank"): [
        {"model": "Logistic Regression",       "accuracy": 0.9738, "precision": 1.0000, "recall": 0.9475, "f1_score": 0.9730, "n_features": 24, "iteration": "Iteration 1 (Full 24)"},
        {"model": "Random Forest",             "accuracy": 0.9831, "precision": 1.0000, "recall": 0.9662, "f1_score": 0.9828, "n_features": 24, "iteration": "Iteration 1 (Full 24)"},
        {"model": "Decision Tree (depth=10)",  "accuracy": 0.9905, "precision": 1.0000, "recall": 0.9810, "f1_score": 0.9904, "n_features": 24, "iteration": "Iteration 1 (Full 24)"},
    ],
}

# --------------------------------------------------------------
# Baseline results from Iteration 2 (reduced 13 features, baseline
# hyperparameters) for three-way comparison in Chapter 4.
# These values are taken directly from feature_selection_eval logs.
# --------------------------------------------------------------
ITER2_WITHIN = {
    "features_phishtank_tranco_dataset": [
        {"model": "Logistic Regression",       "accuracy": 0.9972, "precision": 0.9995, "recall": 0.9950, "f1_score": 0.9972, "n_features": 13, "iteration": "Iteration 2 (Reduced 13)"},
        {"model": "Random Forest",             "accuracy": 0.9978, "precision": 0.9985, "recall": 0.9970, "f1_score": 0.9977, "n_features": 13, "iteration": "Iteration 2 (Reduced 13)"},
        {"model": "Decision Tree (depth=10)",  "accuracy": 0.9980, "precision": 0.9985, "recall": 0.9975, "f1_score": 0.9980, "n_features": 13, "iteration": "Iteration 2 (Reduced 13)"},
    ],
    "features_urlhaus_tranco_dataset": [
        {"model": "Logistic Regression",       "accuracy": 0.9998, "precision": 1.0000, "recall": 0.9995, "f1_score": 0.9997, "n_features": 13, "iteration": "Iteration 2 (Reduced 13)"},
        {"model": "Random Forest",             "accuracy": 0.9998, "precision": 1.0000, "recall": 0.9995, "f1_score": 0.9997, "n_features": 13, "iteration": "Iteration 2 (Reduced 13)"},
        {"model": "Decision Tree (depth=10)",  "accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1_score": 1.0000, "n_features": 13, "iteration": "Iteration 2 (Reduced 13)"},
    ],
}

ITER2_CROSS = {
    ("PhishTank", "URLhaus"): [
        {"model": "Logistic Regression",       "accuracy": 0.9995, "precision": 0.9990, "recall": 1.0000, "f1_score": 0.9995, "n_features": 13, "iteration": "Iteration 2 (Reduced 13)"},
        {"model": "Random Forest",             "accuracy": 1.0000, "precision": 0.9999, "recall": 1.0000, "f1_score": 1.0000, "n_features": 13, "iteration": "Iteration 2 (Reduced 13)"},
        {"model": "Decision Tree (depth=10)",  "accuracy": 0.9996, "precision": 0.9991, "recall": 1.0000, "f1_score": 0.9996, "n_features": 13, "iteration": "Iteration 2 (Reduced 13)"},
    ],
    ("URLhaus", "PhishTank"): [
        {"model": "Logistic Regression",       "accuracy": 0.9809, "precision": 1.0000, "recall": 0.9619, "f1_score": 0.9806, "n_features": 13, "iteration": "Iteration 2 (Reduced 13)"},
        {"model": "Random Forest",             "accuracy": 0.9831, "precision": 1.0000, "recall": 0.9663, "f1_score": 0.9829, "n_features": 13, "iteration": "Iteration 2 (Reduced 13)"},
        {"model": "Decision Tree (depth=10)",  "accuracy": 0.9905, "precision": 1.0000, "recall": 0.9810, "f1_score": 0.9904, "n_features": 13, "iteration": "Iteration 2 (Reduced 13)"},
    ],
}


# --------------------------------------------------------------
# Tee — mirrors all stdout to a file simultaneously
# --------------------------------------------------------------
class _Tee:
    # defines a custom class that duplicates all stdout output to both the terminal and a log file simultaneously
    def __init__(self, file_path: str):
        # initializes the _Tee object by storing the original stdout reference and opening the log file for writing
        self._terminal = sys.stdout                               # saves a reference to the original sys.stdout so terminal output is preserved alongside file output
        self._file     = open(file_path, "w", encoding="utf-8")  # opens the specified log file in write mode with UTF-8 encoding to capture all output

    def write(self, message: str):
        # writes the given message to both the terminal and the log file simultaneously
        self._terminal.write(message)  # writes the message to the terminal using the saved original stdout reference
        self._file.write(message)      # writes the same message to the log file so both outputs remain identical

    def flush(self):
        # flushes both the terminal and file output buffers to ensure all pending output is written immediately
        self._terminal.flush()  # flushes the terminal buffer to force any buffered terminal output to be written immediately
        self._file.flush()      # flushes the file buffer to force any buffered file output to be written immediately, preventing data loss

    def close(self):
        # closes the log file to release the file handle and ensure all data has been written to disk
        self._file.close()  # closes the log file, flushing any remaining buffered content and releasing the file handle


def setup_logging(output_dir: str) -> str:
    # configures logging for the script by redirecting stdout to a _Tee object that writes to both the terminal and a timestamped log file
    os.makedirs(output_dir, exist_ok=True)                                                       # creates the output directory if it does not already exist, using exist_ok=True to avoid errors if it already exists
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")                               # generates a timestamp string in YYYYMMDD_HHMMSS format from the current date and time for use in the log file name
    log_path  = os.path.join(output_dir, f"hyperparameter_tuning_eval_{timestamp}.txt")         # constructs the full path for the timestamped log file by joining the output directory with the formatted filename
    sys.stdout = _Tee(log_path)                                                                  # replaces sys.stdout with a _Tee instance so all subsequent print statements and log messages are written to both the terminal and the log file
    logger  = logging.getLogger()                                                                # retrieves the root logger instance to configure global logging behavior for the entire script
    logger.setLevel(logging.INFO)                                                                # sets the minimum logging level to INFO so that all INFO, WARNING, ERROR, and CRITICAL messages are captured
    handler = logging.StreamHandler(sys.stdout)                                                  # creates a StreamHandler that writes log records to sys.stdout, which is now the _Tee object
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))  # sets the log record format to include time in HH:MM:SS format, log level in brackets, and the message
    logger.addHandler(handler)                                                                   # attaches the configured handler to the root logger so all log calls use this handler
    logging.getLogger(__name__).info("Results will be saved -> %s", os.path.abspath(log_path))  # logs the absolute path of the log file so the user knows where results are being saved
    return log_path  # returns the log file path so the main function can reference it in the final confirmation message


log = logging.getLogger(__name__)  # creates a module-level logger named after the current module for use throughout the script


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    # loads a CSV file into a pandas DataFrame, attempting UTF-8 encoding first and falling back to latin-1 if a UnicodeDecodeError is encountered
    try:
        return pd.read_csv(path, low_memory=False)  # attempts to read the CSV file using default UTF-8 encoding with low_memory=False to avoid dtype inference warnings on large files
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip", low_memory=False)  # if UTF-8 decoding fails, retries with latin-1 encoding and skips any lines that still cause parsing errors


def find_label_column(df: pd.DataFrame) -> str | None:
    # searches the DataFrame column names for a label column by checking each candidate name from LABEL_CANDIDATES in a case-insensitive manner
    lower_map = {c.lower(): c for c in df.columns}  # creates a dictionary mapping each column name lowercased to its original case for case-insensitive lookup
    for key in LABEL_CANDIDATES:                    # iterates through each candidate label column name defined in LABEL_CANDIDATES
        if key in lower_map:                        # checks whether the current candidate name exists in the lowercase column mapping
            return lower_map[key]                   # returns the original-case column name corresponding to the matched candidate
    return None  # returns None if no candidate label column name is found in the DataFrame


def normalize_labels(series: pd.Series) -> pd.Series:
    # normalizes a label series to binary integer values (0 for legitimate, 1 for malicious) regardless of whether the original values are numeric or string-based
    if pd.api.types.is_numeric_dtype(series):           # checks whether the series contains numeric data types which may already be in binary format
        return pd.to_numeric(series, errors="coerce")   # if the series is numeric, converts to numeric type and coerces any non-numeric values to NaN for later filtering
    return series.astype(str).str.strip().str.lower().map(LABEL_MAP)  # if non-numeric, converts to string, strips whitespace, lowercases, and maps each value to 0 or 1 using LABEL_MAP


def prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:
    # loads a feature CSV file, identifies and normalizes the label column, removes rows with invalid labels, and returns the feature matrix X and label vector y
    if not os.path.exists(path):                         # checks whether the specified file path exists before attempting to load it
        log.error("Feature file not found: %s", path)   # logs an error identifying the missing file path
        sys.exit(1)                                      # exits with a non-zero status code to indicate a fatal error caused by the missing input file

    df        = load_dataset(path)       # loads the CSV file into a DataFrame using the load_dataset helper which handles encoding fallback automatically
    label_col = find_label_column(df)    # identifies the name of the label column using the find_label_column helper
    if label_col is None:                # checks whether no recognizable label column was found
        log.error("No label column found in %s", path)  # logs an error identifying the file with no label column
        sys.exit(1)                      # exits with a non-zero status code to indicate a fatal error caused by the missing label column

    y    = normalize_labels(df[label_col])   # normalizes the label column values to binary integers using the normalize_labels helper
    mask = y.isin([0, 1])                    # creates a boolean mask that is True for rows where the normalized label is 0 or 1, filtering out invalid label values
    df   = df[mask].reset_index(drop=True)   # applies the boolean mask to retain only rows with valid labels and resets the integer index to be sequential
    y    = y[mask].reset_index(drop=True).astype(int)  # applies the same mask to the label series, resets the index, and converts values to integer type

    feature_cols = [c for c in df.columns if c not in (label_col, "url")]  # builds a list of feature column names by excluding the label column and the url column
    X = df[feature_cols].select_dtypes(include=["number"])  # creates the feature matrix X by selecting only numeric columns from the identified feature columns

    log.info(
        "Loaded %s: %d rows, %d features | malicious: %d (%.1f%%)  legitimate: %d (%.1f%%)",
        os.path.basename(path), len(X), len(X.columns),  # logs the base filename, total row count, and feature count
        int(y.sum()), 100 * y.mean(),                    # logs the count and percentage of malicious samples
        int((y == 0).sum()), 100 * (1 - y.mean()),       # logs the count and percentage of legitimate samples
    )
    return X, y  # returns the feature matrix X and label vector y as a tuple


# --------------------------------------------------------------
# Grid search
# --------------------------------------------------------------
def run_grid_search(
    X_train:    pd.DataFrame,  # training feature matrix containing only the 13 reduced features selected in Iteration 2
    y_train:    pd.Series,     # training label vector containing binary integer labels corresponding to rows in X_train
    ds_label:   str,           # human-readable dataset name for logging, for example PhishTank or URLhaus
    scaler:     StandardScaler, # fitted StandardScaler instance used to transform features for Logistic Regression grid search
) -> dict:
    # runs GridSearchCV for all three models on the provided training data using 5-fold stratified cross-validation scored by F1, and returns a dictionary of best estimator objects and their best parameter configurations
    """
    Run GridSearchCV for all three models on the reduced feature set.
    Returns a dict mapping model name to (best_estimator, best_params, best_score).
    Scoring metric: F1-score (consistent with project evaluation methodology).
    CV strategy: 5-fold stratified (consistent with Iterations 1 and 2).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # creates a 5-fold stratified cross-validation splitter with shuffling enabled and a fixed random seed for reproducibility, preserving class distribution across all folds

    best_models = {}  # initializes an empty dictionary to store the best estimator, best parameters, and best cross-validation score for each model after grid search completes

    # ----------------------------------------------------------
    # Logistic Regression grid search
    # Uses scaled features because LR is sensitive to feature magnitudes.
    # Scaler is already fitted on training data by the caller.
    # ----------------------------------------------------------
    log.info("  Running grid search: Logistic Regression on %s...", ds_label)  # logs that the Logistic Regression grid search is beginning for this dataset
    X_tr_sc = scaler.transform(X_train)  # transforms the training features using the already-fitted scaler without refitting, applying the same scaling parameters learned from the training data

    lr_grid = GridSearchCV(
        LogisticRegression(random_state=42, class_weight="balanced", max_iter=2000),  # instantiates the base Logistic Regression estimator with a fixed random seed, balanced class weights, and increased max_iter to ensure convergence across all parameter combinations
        PARAM_GRIDS["Logistic Regression"],  # passes the C parameter grid defining the regularization strengths to be exhaustively searched
        cv=cv,            # uses the 5-fold stratified cross-validation splitter defined above
        scoring="f1",     # optimizes for F1-score consistent with the project's evaluation methodology and emphasis on balancing precision and recall
        n_jobs=-1,        # uses all available CPU cores for parallel grid search to reduce computation time
        refit=True,       # automatically refits the best estimator on the full training set after grid search completes, making it ready for direct prediction
    )
    lr_grid.fit(X_tr_sc, y_train)  # runs the full grid search on the scaled training features, training and evaluating every parameter combination across all 5 folds
    best_models["Logistic Regression"] = (lr_grid.best_estimator_, lr_grid.best_params_, lr_grid.best_score_)  # stores the best fitted estimator, best parameter dictionary, and best cross-validation F1-score for Logistic Regression
    log.info("    Best params: %s  |  CV F1: %.4f", lr_grid.best_params_, lr_grid.best_score_)  # logs the best hyperparameter configuration and corresponding cross-validation F1-score found by grid search

    # ----------------------------------------------------------
    # Random Forest grid search
    # Uses unscaled features as RF is invariant to feature magnitude.
    # ----------------------------------------------------------
    log.info("  Running grid search: Random Forest on %s...", ds_label)  # logs that the Random Forest grid search is beginning for this dataset
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),  # instantiates the base Random Forest estimator with a fixed random seed, balanced class weights, and parallel tree building
        PARAM_GRIDS["Random Forest"],  # passes the n_estimators and max_depth parameter grid to be exhaustively searched across all combinations
        cv=cv,         # uses the same 5-fold stratified cross-validation splitter for consistency
        scoring="f1",  # optimizes for F1-score
        n_jobs=-1,     # uses all available CPU cores for parallel grid search
        refit=True,    # automatically refits the best estimator on the full training set after grid search
    )
    rf_grid.fit(X_train, y_train)  # runs the full grid search on the unscaled training features across all parameter combinations and folds
    best_models["Random Forest"] = (rf_grid.best_estimator_, rf_grid.best_params_, rf_grid.best_score_)  # stores the best fitted estimator, parameters, and score for Random Forest
    log.info("    Best params: %s  |  CV F1: %.4f", rf_grid.best_params_, rf_grid.best_score_)  # logs the best parameters and cross-validation score found for Random Forest

    # ----------------------------------------------------------
    # Decision Tree grid search
    # Uses unscaled features as DT is invariant to feature magnitude.
    # ----------------------------------------------------------
    log.info("  Running grid search: Decision Tree on %s...", ds_label)  # logs that the Decision Tree grid search is beginning for this dataset
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),  # instantiates the base Decision Tree estimator with a fixed random seed and balanced class weights
        PARAM_GRIDS["Decision Tree"],  # passes the max_depth and min_samples_split parameter grid to be exhaustively searched
        cv=cv,         # uses the same 5-fold stratified cross-validation splitter
        scoring="f1",  # optimizes for F1-score
        n_jobs=-1,     # uses all available CPU cores for parallel grid search
        refit=True,    # automatically refits the best estimator on the full training set after grid search
    )
    dt_grid.fit(X_train, y_train)  # runs the full grid search on the unscaled training features across all parameter combinations and folds
    best_models["Decision Tree (depth=10)"] = (dt_grid.best_estimator_, dt_grid.best_params_, dt_grid.best_score_)  # stores the best fitted estimator, parameters, and score using the same model name key as Iterations 1 and 2 for consistent comparison
    log.info("    Best params: %s  |  CV F1: %.4f", dt_grid.best_params_, dt_grid.best_score_)  # logs the best parameters and cross-validation score found for Decision Tree

    return best_models  # returns the dictionary mapping each model name to its best estimator, best parameters, and best cross-validation F1-score


# --------------------------------------------------------------
# Evaluate tuned models
# --------------------------------------------------------------
def evaluate_tuned_models(
    X_train:      pd.DataFrame,  # training feature matrix using the 13 reduced features
    y_train:      pd.Series,     # training label vector with binary integer labels
    X_test:       pd.DataFrame,  # test feature matrix using the same 13 reduced features
    y_test:       pd.Series,     # test label vector with binary integer labels
    best_models:  dict,          # dictionary mapping model names to (best_estimator, best_params, best_score) tuples from grid search
    train_name:   str,           # name of the training dataset for result dictionary labeling
    test_name:    str,           # name of the test dataset for result dictionary labeling
    results_list: list,          # shared list to which result dictionaries are appended for aggregation
    scaler:       StandardScaler, # StandardScaler fitted on training data, used to scale features for Logistic Regression evaluation
) -> list:
    # evaluates all three tuned models on the provided test data using their best estimators from grid search, computes all metrics, and returns a list of result dictionaries
    local_results = []  # initializes an empty list to store result dictionaries for this specific evaluation call

    X_tr_sc = scaler.transform(X_train)  # transforms the training features using the fitted scaler for Logistic Regression evaluation, applying transform only without refitting to prevent data leakage
    X_te_sc = scaler.transform(X_test)   # transforms the test features using the same fitted scaler parameters to ensure consistent scaling between training and test data

    for model_name, (best_est, best_params, best_cv_score) in best_models.items():
        # iterates over each model name and its corresponding best estimator, best parameters, and best cross-validation score from the grid search results

        if model_name == "Logistic Regression":
            X_tr_eval = X_tr_sc   # uses the scaled training features for Logistic Regression evaluation since it requires normalized feature magnitudes
            X_te_eval = X_te_sc   # uses the scaled test features for Logistic Regression evaluation
        else:
            X_tr_eval = X_train   # uses the unscaled training features for Random Forest and Decision Tree which are invariant to feature magnitude
            X_te_eval = X_test    # uses the unscaled test features for tree-based models

        best_est.fit(X_tr_eval, y_train)  # refits the best estimator on the full training set using the appropriate scaled or unscaled features to ensure it has been trained on all available training data before evaluation
        y_pred = best_est.predict(X_te_eval)  # generates predicted class labels for the test set using the refitted best estimator
        cm     = confusion_matrix(y_test, y_pred)  # computes the confusion matrix comparing true test labels to predicted labels, producing a 2x2 matrix of TN, FP, FN, TP counts

        r = {
            "iteration":   "Iteration 3 (Tuned 13)",  # labels this result as Iteration 3 for three-way comparison against Iterations 1 and 2
            "train_on":    train_name,                 # stores the training dataset name for result identification
            "test_on":     test_name,                  # stores the test dataset name for result identification
            "model":       model_name,                 # stores the model name for result identification
            "best_params": str(best_params),           # stores the best hyperparameter configuration as a string for logging and CSV saving
            "cv_f1":       round(best_cv_score, 4),    # stores the best cross-validation F1-score rounded to 4 decimal places
            "accuracy":    round(accuracy_score(y_test, y_pred),  4),                          # computes overall accuracy rounded to 4 decimal places
            "precision":   round(precision_score(y_test, y_pred, zero_division=0), 4),         # computes precision rounded to 4 decimal places with zero_division=0 to handle edge cases
            "recall":      round(recall_score(y_test, y_pred,    zero_division=0), 4),         # computes recall rounded to 4 decimal places with zero_division=0 to handle edge cases
            "f1_score":    round(f1_score(y_test, y_pred,        zero_division=0), 4),         # computes F1-score rounded to 4 decimal places with zero_division=0 to handle edge cases
            "tn": int(cm[0][0]), "fp": int(cm[0][1]),  # extracts true negative and false positive counts from the confusion matrix and converts to integer
            "fn": int(cm[1][0]), "tp": int(cm[1][1]),  # extracts false negative and true positive counts from the confusion matrix and converts to integer
            "n_features":  X_tr_eval.shape[1],         # stores the number of features used in this evaluation, should always be 13 for Iteration 3
        }
        local_results.append(r)   # appends the result dictionary to the local results list for this evaluation call
        results_list.append(r)    # appends the same result dictionary to the shared results list for aggregation across all evaluations

        print(f"\n  {model_name}")
        print(f"    Best params : {best_params}")
        print(f"    CV F1       : {best_cv_score:.4f}")
        print(f"    Accuracy    : {r['accuracy']:.4f}")
        print(f"    Precision   : {r['precision']:.4f}")
        print(f"    Recall      : {r['recall']:.4f}")
        print(f"    F1-score    : {r['f1_score']:.4f}")
        print(classification_report(
            y_test, y_pred,
            target_names=["Legitimate", "Malicious"],
            zero_division=0
        ))  # prints the full per-class classification report showing precision, recall, F1-score, and support for both the legitimate and malicious classes

    return local_results  # returns the list of result dictionaries for this evaluation call


# --------------------------------------------------------------
# Print three-way comparison
# --------------------------------------------------------------
def print_three_way_comparison(
    scenario:   str,    # descriptive label for the evaluation scenario being compared, for example Within-dataset PhishTank
    iter1_res:  list,   # list of result dictionaries from Iteration 1 for this scenario
    iter2_res:  list,   # list of result dictionaries from Iteration 2 for this scenario
    iter3_res:  list,   # list of result dictionaries from Iteration 3 for this scenario
):
    # prints a formatted three-way comparison table showing Iteration 1, 2, and 3 results side by side with delta F1 annotations relative to the Iteration 1 baseline
    models = ["Logistic Regression", "Random Forest", "Decision Tree (depth=10)"]  # defines the ordered list of model names for the comparison table

    print(f"\n{'='*80}")
    print(f"THREE-WAY COMPARISON — {scenario}")
    print(f"{'='*80}")
    header = f"  {'Model':<32s} {'Iteration':>22s} {'Acc':>8s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s}"
    print(header)
    print("  " + "-" * 78)

    i1_map = {r["model"]: r for r in iter1_res}  # creates a dictionary mapping model names to Iteration 1 result dictionaries for quick lookup
    i2_map = {r["model"]: r for r in iter2_res}  # creates a dictionary mapping model names to Iteration 2 result dictionaries for quick lookup
    i3_map = {r["model"]: r for r in iter3_res}  # creates a dictionary mapping model names to Iteration 3 result dictionaries for quick lookup

    for m in models:                              # iterates over each model in the defined order
        r1 = i1_map.get(m)                        # retrieves the Iteration 1 result for this model
        r2 = i2_map.get(m)                        # retrieves the Iteration 2 result for this model
        r3 = i3_map.get(m)                        # retrieves the Iteration 3 result for this model
        baseline_f1 = r1["f1_score"] if r1 else 0 # stores the Iteration 1 F1-score as the baseline for delta computation

        if r1:
            print(f"  {m:<32s} {'Iter 1 (Full 24)':>22s} "
                  f"{r1['accuracy']:>8.4f} {r1['precision']:>8.4f} "
                  f"{r1['recall']:>8.4f} {r1['f1_score']:>8.4f}")  # prints the Iteration 1 row for this model with all four metrics
        if r2:
            d2 = r2["f1_score"] - baseline_f1     # computes the F1-score delta between Iteration 2 and the Iteration 1 baseline
            s2 = "+" if d2 >= 0 else ""            # prepends a plus sign to non-negative deltas
            print(f"  {m:<32s} {'Iter 2 (Reduced 13)':>22s} "
                  f"{r2['accuracy']:>8.4f} {r2['precision']:>8.4f} "
                  f"{r2['recall']:>8.4f} {r2['f1_score']:>8.4f}  (ΔF1={s2}{d2:.4f})")  # prints the Iteration 2 row with delta F1 relative to Iteration 1
        if r3:
            d3 = r3["f1_score"] - baseline_f1     # computes the F1-score delta between Iteration 3 and the Iteration 1 baseline
            s3 = "+" if d3 >= 0 else ""            # prepends a plus sign to non-negative deltas
            print(f"  {m:<32s} {'Iter 3 (Tuned 13)':>22s} "
                  f"{r3['accuracy']:>8.4f} {r3['precision']:>8.4f} "
                  f"{r3['recall']:>8.4f} {r3['f1_score']:>8.4f}  (ΔF1={s3}{d3:.4f})")  # prints the Iteration 3 row with delta F1 relative to Iteration 1
        print()  # prints a blank line between models for visual separation


# --------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------
def _three_way_bar_ax(ax, iter1_res, iter2_res, iter3_res, title):
    # renders a grouped bar chart on the provided axes comparing F1-scores across all three iterations for each model
    models  = [r["model"].replace(" (depth=10)", "\n(depth=10)") for r in iter1_res]  # extracts model names and formats the depth notation with a newline for x-axis readability
    x       = np.arange(len(models))  # creates an array of x positions, one per model group
    width   = 0.22  # defines the width of each bar so three bars fit side by side per model group without overlapping

    i1_f1 = [r["f1_score"] for r in iter1_res]  # extracts Iteration 1 F1-scores for bar heights
    i2_f1 = [r["f1_score"] for r in iter2_res]  # extracts Iteration 2 F1-scores for bar heights
    i3_f1 = [r["f1_score"] for r in iter3_res]  # extracts Iteration 3 F1-scores for bar heights

    b1 = ax.bar(x - width, i1_f1, width, label="Iter 1 — Full 24",    color=METRIC_COLOURS[0], alpha=0.88)  # draws Iteration 1 bars in blue positioned to the left of each x position
    b2 = ax.bar(x,         i2_f1, width, label="Iter 2 — Reduced 13", color=METRIC_COLOURS[1], alpha=0.88)  # draws Iteration 2 bars in green centered at each x position
    b3 = ax.bar(x + width, i3_f1, width, label="Iter 3 — Tuned 13",   color=METRIC_COLOURS[2], alpha=0.88)  # draws Iteration 3 bars in orange positioned to the right of each x position

    for bars, vals in [(b1, i1_f1), (b2, i2_f1), (b3, i3_f1)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=6, color="white")  # adds a value label centered above each bar showing the F1-score to 4 decimal places

    all_vals = i1_f1 + i2_f1 + i3_f1  # combines all F1 values to determine the y-axis lower bound
    ax.set_xticks(x)                                                  # sets x-axis tick positions to align with model group centers
    ax.set_xticklabels(models, color="white", fontsize=8)             # sets x-axis tick labels to model names in white text
    ax.set_ylim(min(all_vals) - 0.04, 1.08)                          # sets y-axis limits with padding below the minimum and above 1.0 for value labels
    ax.set_ylabel("F1-Score", color="white", fontsize=9)              # labels the y-axis
    ax.set_title(title, color="white", fontweight="bold", fontsize=9) # sets the subplot title
    ax.tick_params(colors="white")                                    # sets tick colors to white
    ax.spines[:].set_color("#444")                                    # sets spine border colors to dark gray
    ax.legend(loc="lower right", fontsize=7, facecolor="#21262D", labelcolor="white")  # adds a legend identifying each iteration's bars


def _recall_delta_ax(ax, iter1_res, iter3_res, title):
    # renders a horizontal bar chart showing the recall change between Iteration 3 tuned models and the Iteration 1 baseline for each model
    models  = [r["model"].replace(" (depth=10)", "\n(depth=10)") for r in iter1_res]  # extracts and formats model names for y-axis labels
    i1_rec  = {r["model"]: r["recall"] for r in iter1_res}  # creates a lookup dictionary of Iteration 1 recall values by model name
    i3_rec  = {r["model"]: r["recall"] for r in iter3_res}  # creates a lookup dictionary of Iteration 3 recall values by model name
    deltas  = [i3_rec[r["model"]] - i1_rec[r["model"]] for r in iter1_res]  # computes recall delta as Iteration 3 minus Iteration 1 for each model
    colours = ["#2EA043" if d >= 0 else "#C0392B" for d in deltas]  # assigns green to improvements and red to degradations

    y    = np.arange(len(models))  # creates y positions for the horizontal bars
    bars = ax.barh(y, deltas, color=colours, alpha=0.85)  # draws horizontal bars with heights equal to recall deltas and colors based on direction
    for bar, val in zip(bars, deltas):
        sign = "+" if val >= 0 else ""  # prepends plus sign to non-negative deltas
        ax.text(bar.get_width() + (0.0002 if val >= 0 else -0.0002),
                bar.get_y() + bar.get_height()/2,
                f"{sign}{val:.4f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8, color="white")  # places delta label at the end of each bar
    ax.set_yticks(y)                                                           # sets y-axis tick positions
    ax.set_yticklabels(models, color="white", fontsize=8)                      # sets y-axis tick labels to model names in white
    ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")                 # draws a vertical reference line at zero to separate improvements from degradations
    ax.set_xlabel("ΔRecall (Iteration 3 − Iteration 1)", color="white", fontsize=9)  # labels the x-axis
    ax.set_title(title, color="white", fontweight="bold", fontsize=9)          # sets the subplot title
    ax.tick_params(colors="white")                                             # sets tick colors to white
    ax.spines[:].set_color("#444")                                             # sets spine border colors to dark gray


def _metrics_table_ax(ax, results, title):
    # renders a compact metrics table on the provided axes showing model name, best params, accuracy, precision, recall, F1-score, and CV F1 for each result
    ax.axis("off")           # turns off axes since this subplot displays a table
    ax.set_facecolor("#161B22")  # sets background to dark for dashboard consistency

    table_data = [
        [r["model"],
         r.get("best_params", "baseline"),  # displays best parameters if available, or baseline if this is an Iteration 1 result stored without best_params
         f"{r['accuracy']:.4f}",
         f"{r['precision']:.4f}",
         f"{r['recall']:.4f}",
         f"{r['f1_score']:.4f}",
         f"{r.get('cv_f1', '-')}"]  # displays cross-validation F1 if available, or a dash for Iteration 1 baseline results
        for r in results
    ]
    col_labels = ["Model", "Best Params", "Acc", "Prec", "Rec", "F1", "CV F1"]  # defines the column header labels for the metrics table
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="center", loc="center")  # creates the matplotlib table with data rows and column headers, center-aligned
    tbl.auto_set_font_size(False)  # disables automatic font size adjustment
    tbl.set_fontsize(7)            # sets font size to 7 for compact display given the additional columns
    tbl.scale(1, 1.8)              # scales rows to 1.8x height for readability

    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#C0392B")                          # sets header cell backgrounds to red
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")  # sets header text to bold white

    row_colours = ["#1C2128", "#21262D"]  # alternating row colors for striped effect
    for i in range(len(table_data)):
        for j in range(len(col_labels)):
            tbl[(i+1, j)].set_facecolor(row_colours[i % 2])  # applies alternating row colors
            tbl[(i+1, j)].set_text_props(color="white")       # sets data cell text to white

    ax.set_title(title, color="white", fontsize=8, fontweight="bold", pad=6)  # sets the table subplot title


def save_dashboards(
    within_i1, within_i2, within_i3,  # within-dataset results dictionaries for Iterations 1, 2, and 3 keyed by dataset name
    cross_i1,  cross_i2,  cross_i3,   # cross-dataset results dictionaries for Iterations 1, 2, and 3 keyed by (train, test) tuples
    output_dir: str,                   # directory path where dashboard PNG files will be saved
):
    # generates and saves two dashboard PNG files: one for within-dataset three-way comparison and one for cross-dataset three-way comparison
    sns.set_theme(style="whitegrid", font_scale=0.85)  # sets the seaborn theme with whitegrid style and reduced font scale for compact dashboard layout

    # ── Dashboard 1: Within-dataset ──────────────────────────────
    fig1 = plt.figure(figsize=(24, 16))      # creates the within-dataset dashboard figure with generous dimensions
    fig1.patch.set_facecolor("#0D1117")      # sets the figure background to a very dark shade for the dashboard theme
    fig1.suptitle(
        "Iteration 3 — Hyperparameter Tuning: Within-Dataset Evaluation\n"
        "Three-Way Comparison: Iteration 1 (Full 24) vs Iteration 2 (Reduced 13) vs Iteration 3 (Tuned 13)",
        fontsize=12, fontweight="bold", color="white", y=0.98  # sets the figure title describing all three iterations
    )
    gs1 = gridspec.GridSpec(2, 4, figure=fig1,
                            hspace=0.55, wspace=0.45,
                            left=0.05, right=0.97, top=0.91, bottom=0.05)  # creates a 2-row by 4-column grid layout for the within-dataset dashboard

    dataset_keys = list(within_i1.keys())  # retrieves dataset name keys for iteration

    # Row 0: Three-way F1 comparison bars
    for col_idx, ds_name in enumerate(dataset_keys[:2]):
        ds_label = ds_name.replace("features_", "").replace("_", " ").title()  # formats dataset name for display
        ax_bar   = fig1.add_subplot(gs1[0, col_idx * 2 : col_idx * 2 + 2])    # creates subplot spanning 2 columns for this dataset's bar chart
        ax_bar.set_facecolor("#161B22")  # sets background to dark
        _three_way_bar_ax(ax_bar,
                          within_i1[ds_name],
                          within_i2[ds_name],
                          within_i3[ds_name],
                          f"F1 Three-Way Comparison — {ds_label}")  # renders the three-way bar chart for this dataset

    # Row 1: Metrics tables for Iteration 3 results
    for col_idx, ds_name in enumerate(dataset_keys[:2]):
        ds_label = ds_name.replace("features_", "").replace("_", " ").title()
        ax_tbl   = fig1.add_subplot(gs1[1, col_idx * 2 : col_idx * 2 + 2])  # creates subplot spanning 2 columns for this dataset's metrics table
        ax_tbl.set_facecolor("#161B22")
        _metrics_table_ax(ax_tbl, within_i3[ds_name],
                          f"Iteration 3 Best Params — {ds_label}")  # renders the Iteration 3 metrics table with best parameter configurations

    save_path1 = os.path.join(output_dir, "dashboard__hyperparameter_tuning_within.png")  # constructs the output file path for the within-dataset dashboard
    plt.savefig(save_path1, dpi=150, bbox_inches="tight", facecolor=fig1.get_facecolor())  # saves the dashboard as PNG at 150 DPI with tight bounding box
    log.info("Dashboard saved -> %s", os.path.abspath(save_path1))  # logs the absolute path of the saved dashboard
    plt.close(fig1)  # closes the figure to release memory

    # ── Dashboard 2: Cross-dataset ───────────────────────────────
    fig2 = plt.figure(figsize=(24, 16))      # creates the cross-dataset dashboard figure
    fig2.patch.set_facecolor("#0D1117")      # sets the dark background
    fig2.suptitle(
        "Iteration 3 — Hyperparameter Tuning: Cross-Dataset Evaluation\n"
        "Three-Way Comparison: Iteration 1 (Full 24) vs Iteration 2 (Reduced 13) vs Iteration 3 (Tuned 13)",
        fontsize=12, fontweight="bold", color="white", y=0.98  # sets the cross-dataset dashboard title
    )
    gs2 = gridspec.GridSpec(3, 4, figure=fig2,
                            hspace=0.55, wspace=0.45,
                            left=0.05, right=0.97, top=0.91, bottom=0.05)  # creates a 3-row by 4-column grid for the cross-dataset dashboard

    cross_keys = list(cross_i1.keys())  # retrieves cross-evaluation direction keys

    # Row 0: Three-way F1 comparison bars
    for col_idx, key in enumerate(cross_keys[:2]):
        train_n, test_n = key  # unpacks the direction tuple
        ax_bar = fig2.add_subplot(gs2[0, col_idx * 2 : col_idx * 2 + 2])  # creates subplot for this direction's bar chart
        ax_bar.set_facecolor("#161B22")
        _three_way_bar_ax(ax_bar,
                          cross_i1[key],
                          cross_i2[key],
                          cross_i3[key],
                          f"F1: Train {train_n} → Test {test_n}")  # renders the three-way F1 bar chart for this cross-evaluation direction

    # Row 1: Recall delta charts (Iteration 3 vs Iteration 1 baseline)
    for col_idx, key in enumerate(cross_keys[:2]):
        train_n, test_n = key
        ax_delta = fig2.add_subplot(gs2[1, col_idx * 2 : col_idx * 2 + 2])  # creates subplot for the recall delta chart
        ax_delta.set_facecolor("#161B22")
        _recall_delta_ax(ax_delta,
                         cross_i1[key],
                         cross_i3[key],
                         f"ΔRecall vs Iter 1: Train {train_n} → Test {test_n}")  # renders the recall delta chart comparing Iteration 3 to Iteration 1 baseline

    # Row 2: Iteration 3 metrics tables
    for col_idx, key in enumerate(cross_keys[:2]):
        train_n, test_n = key
        ax_tbl = fig2.add_subplot(gs2[2, col_idx * 2 : col_idx * 2 + 2])  # creates subplot for the cross-dataset metrics table
        ax_tbl.set_facecolor("#161B22")
        _metrics_table_ax(ax_tbl, cross_i3[key],
                          f"Iter 3 Best Params — Train {train_n} → Test {test_n}")  # renders the Iteration 3 cross-dataset metrics table

    save_path2 = os.path.join(output_dir, "dashboard__hyperparameter_tuning_cross.png")  # constructs the output file path for the cross-dataset dashboard
    plt.savefig(save_path2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())  # saves the cross-dataset dashboard as PNG
    log.info("Dashboard saved -> %s", os.path.abspath(save_path2))  # logs the absolute path of the saved dashboard
    plt.close(fig2)  # closes the figure to release memory


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main(test_size: float = 0.2) -> None:
    # orchestrates the full Iteration 3 hyperparameter tuning evaluation: loads datasets, applies the Iteration 2 reduced feature set, runs grid search for all models on both datasets, evaluates tuned models on within-dataset and cross-dataset directions, produces three-way comparisons, and saves all results and dashboards
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # creates the outputs directory if it does not already exist
    log_path    = setup_logging(OUTPUT_DIR)  # configures logging and stdout redirection, returning the log file path
    all_results = []      # initializes the shared list to accumulate all Iteration 3 result dictionaries
    best_params_log = []  # initializes a list to record best hyperparameter configurations for saving to best_params.csv

    # ----------------------------------------------------------
    # 1. Load datasets
    # ----------------------------------------------------------
    log.info("Loading datasets...")  # logs the start of the dataset loading phase
    X_phish, y_phish = prepare(PHISHTANK_FILE)  # loads the PhishTank feature dataset returning the full feature matrix and label vector
    X_url,   y_url   = prepare(URLHAUS_FILE)    # loads the URLhaus feature dataset returning the full feature matrix and label vector

    # Apply Iteration 2 reduced feature set
    log.info("Applying Iteration 2 reduced feature set (%d features)...", len(REDUCED_FEATURES))  # logs the number of features being applied from Iteration 2
    missing_phish = [f for f in REDUCED_FEATURES if f not in X_phish.columns]  # identifies any reduced features that are absent from the PhishTank feature matrix
    missing_url   = [f for f in REDUCED_FEATURES if f not in X_url.columns]    # identifies any reduced features that are absent from the URLhaus feature matrix
    if missing_phish or missing_url:
        log.error("Missing features in datasets: PhishTank=%s  URLhaus=%s", missing_phish, missing_url)  # logs an error if any reduced features are missing from either dataset
        sys.exit(1)  # exits if required features are not found, preventing evaluation with an incomplete feature set

    X_phish_r = X_phish[REDUCED_FEATURES]  # creates the reduced PhishTank feature matrix by selecting only the 13 features that survived the Iteration 2 threshold cut
    X_url_r   = X_url[REDUCED_FEATURES]    # creates the reduced URLhaus feature matrix by selecting the same 13 features

    # ----------------------------------------------------------
    # 2. Within-dataset grid search and evaluation
    # ----------------------------------------------------------
    log.info("=" * 60)
    log.info("WITHIN-DATASET GRID SEARCH AND EVALUATION")
    log.info("=" * 60)

    within_i3 = {}  # initializes a dictionary to store Iteration 3 within-dataset results keyed by dataset name

    for ds_name, X_full_r, y_full in [
        ("features_phishtank_tranco_dataset", X_phish_r, y_phish),  # defines PhishTank reduced dataset entry
        ("features_urlhaus_tranco_dataset",   X_url_r,   y_url),    # defines URLhaus reduced dataset entry
    ]:
        ds_label = ds_name.replace("features_", "").replace("_tranco_dataset", "").title()  # formats the dataset name for display
        log.info("\n--- Grid Search: %s ---", ds_label)  # logs the start of grid search for this dataset

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_full_r, y_full,
            test_size=test_size, random_state=42, stratify=y_full
        )  # splits the reduced feature dataset into 80/20 training and test sets with stratification and a fixed random seed matching Iterations 1 and 2

        scaler = StandardScaler()       # instantiates a new StandardScaler for this dataset
        scaler.fit(X_tr)                # fits the scaler exclusively on the training data to prevent data leakage, learning mean and standard deviation from training examples only

        best_models = run_grid_search(X_tr, y_tr, ds_label, scaler)  # runs grid search for all three models on this dataset's training data, returning best estimators and parameters

        for model_name, (best_est, best_params, best_cv) in best_models.items():
            best_params_log.append({
                "dataset":    ds_label,       # records which dataset this best configuration was found on
                "model":      model_name,     # records the model name
                "best_params": str(best_params),  # records the best parameter dictionary as a string
                "cv_f1":      round(best_cv, 4),  # records the best cross-validation F1-score
            })  # appends a record of the best parameters for this model and dataset to the log list for saving to best_params.csv

        log.info("\nEvaluating tuned models: %s within-dataset...", ds_label)  # logs the start of within-dataset evaluation using tuned models
        print(f"\n{'='*60}")
        print(f"WITHIN-DATASET EVALUATION — {ds_label} — Iteration 3 (Tuned)")
        print(f"{'='*60}")

        i3_results = evaluate_tuned_models(
            X_tr, y_tr, X_te, y_te,
            best_models, ds_label, ds_label,
            all_results, scaler
        )  # evaluates all three tuned models on the within-dataset test set and appends results to all_results
        within_i3[ds_name] = i3_results  # stores the Iteration 3 within-dataset results for this dataset

        i1 = ITER1_WITHIN.get(ds_name, [])  # retrieves the Iteration 1 baseline results for this dataset from the hardcoded constants
        i2 = ITER2_WITHIN.get(ds_name, [])  # retrieves the Iteration 2 results for this dataset from the hardcoded constants
        print_three_way_comparison(f"Within-dataset — {ds_label}", i1, i2, i3_results)  # prints the three-way comparison table for this dataset

    # ----------------------------------------------------------
    # 3. Cross-dataset grid search and evaluation
    # ----------------------------------------------------------
    log.info("=" * 60)
    log.info("CROSS-DATASET EVALUATION WITH TUNED MODELS")
    log.info("=" * 60)

    cross_i3 = {}  # initializes a dictionary to store Iteration 3 cross-dataset results keyed by (train_name, test_name) tuples

    for train_name, test_name, X_tr_full, y_tr, X_te_full, y_te in [
        ("PhishTank", "URLhaus",   X_phish_r, y_phish, X_url_r,   y_url),    # defines Direction 1: train on reduced PhishTank, test on reduced URLhaus
        ("URLhaus",   "PhishTank", X_url_r,   y_url,   X_phish_r, y_phish),  # defines Direction 2: train on reduced URLhaus, test on reduced PhishTank
    ]:
        key = (train_name, test_name)  # creates the dictionary key as a direction tuple
        log.info("\n--- Cross-eval: Train %s → Test %s ---", train_name, test_name)  # logs the start of cross-dataset evaluation for this direction

        scaler_cross = StandardScaler()    # instantiates a new StandardScaler for this cross-evaluation direction
        scaler_cross.fit(X_tr_full)        # fits the scaler exclusively on the training dataset to prevent data leakage across the cross-evaluation boundary

        log.info("  Running grid search on training data (%s)...", train_name)  # logs that grid search is being run on the training dataset for this direction
        best_models_cross = run_grid_search(X_tr_full, y_tr, train_name, scaler_cross)  # runs grid search using only the training dataset for this direction

        print(f"\n{'='*60}")
        print(f"CROSS-DATASET: Train {train_name} → Test {test_name} — Iteration 3 (Tuned)")
        print(f"{'='*60}")

        i3_cross_results = evaluate_tuned_models(
            X_tr_full, y_tr, X_te_full, y_te,
            best_models_cross, train_name, test_name,
            all_results, scaler_cross
        )  # evaluates all three tuned models on the cross-dataset test set and appends results to all_results
        cross_i3[key] = i3_cross_results  # stores the Iteration 3 cross-dataset results for this direction

        i1_cross = ITER1_CROSS.get(key, [])  # retrieves Iteration 1 cross-dataset results for this direction
        i2_cross = ITER2_CROSS.get(key, [])  # retrieves Iteration 2 cross-dataset results for this direction
        print_three_way_comparison(
            f"Cross-dataset — Train {train_name} → Test {test_name}",
            i1_cross, i2_cross, i3_cross_results
        )  # prints the three-way comparison table for this cross-evaluation direction

    # ----------------------------------------------------------
    # 4. Save results
    # ----------------------------------------------------------
    results_df = pd.DataFrame(all_results)  # converts the accumulated Iteration 3 result dictionaries into a DataFrame
    results_path = os.path.join(OUTPUT_DIR, "hyperparameter_tuning_results.csv")  # constructs the output file path for the results CSV
    results_df.to_csv(results_path, index=False)  # saves the results DataFrame to CSV without the index column
    log.info("Results saved -> %s", os.path.abspath(results_path))  # logs the absolute path of the saved results CSV

    params_df   = pd.DataFrame(best_params_log)  # converts the best parameters log list into a DataFrame
    params_path = os.path.join(OUTPUT_DIR, "best_params.csv")  # constructs the output file path for the best parameters CSV
    params_df.to_csv(params_path, index=False)  # saves the best parameters DataFrame to CSV
    log.info("Best params saved -> %s", os.path.abspath(params_path))  # logs the absolute path of the saved best parameters CSV

    # ----------------------------------------------------------
    # 5. Print full summary
    # ----------------------------------------------------------
    print(f"\n{'='*80}")
    print("FULL ITERATION 3 RESULTS SUMMARY")
    print(f"{'='*80}")
    print(results_df[[
        "iteration", "train_on", "test_on", "model",
        "accuracy", "precision", "recall", "f1_score", "cv_f1", "n_features"
    ]].to_string(index=False))  # prints the full Iteration 3 results table with all relevant columns formatted without the index

    print(f"\n{'='*80}")
    print("BEST HYPERPARAMETER CONFIGURATIONS")
    print(f"{'='*80}")
    print(params_df.to_string(index=False))  # prints all best parameter configurations for every model and dataset combination

    # ----------------------------------------------------------
    # 6. Dashboards
    # ----------------------------------------------------------
    save_dashboards(
        ITER1_WITHIN, ITER2_WITHIN, within_i3,
        ITER1_CROSS,  ITER2_CROSS,  cross_i3,
        OUTPUT_DIR
    )  # generates and saves the within-dataset and cross-dataset three-way comparison dashboard PNG files

    print("\nHyperparameter tuning evaluation complete.")  # prints completion message

    if isinstance(sys.stdout, _Tee):
        sys.stdout.close()          # closes the log file to flush all remaining buffered output to disk
        sys.stdout = sys.__stdout__ # restores sys.stdout to the original terminal output
    print(f"\nFull results saved -> {os.path.abspath(log_path)}")  # prints the log file path after stdout has been restored


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------
def parse_args():
    # parses command-line arguments to allow the user to customize the test set size without modifying the script
    p = argparse.ArgumentParser(
        description="Iteration 3: Grid search hyperparameter tuning on the Iteration 2 reduced feature set."
    )
    p.add_argument(
        "--testsize", type=float, default=0.2,
        help="Test set fraction for within-dataset split (default: 0.2)"  # defines the --testsize argument defaulting to 0.2, matching the test size used in Iterations 1 and 2
    )
    return p.parse_args()  # parses command-line arguments and returns them as a Namespace object


if __name__ == "__main__":
    # checks whether the script is being run directly and if so parses arguments and calls the main function
    try:
        args = parse_args()                # parses command-line arguments to retrieve the test size value
        main(test_size=args.testsize)      # calls the main function with the parsed test size to execute the full Iteration 3 hyperparameter tuning evaluation
    except Exception:
        log.error("Unhandled error:\n%s", traceback.format_exc())  # catches any unhandled exception and logs the full traceback for diagnosis
        sys.exit(1)  # exits with non-zero status code to indicate the script terminated due to an error

