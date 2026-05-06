# -*- coding: utf-8 -*-
# --------------------------------------------------------------
# src/feature_selection_eval.py
# --------------------------------------------------------------
# Purpose:
#   Iteration 2 — Feature Selection and Re-evaluation.
#
#   This script answers the question:
#   "Does removing low-importance features improve generalisation
#    across datasets, and does it affect within-dataset accuracy?"
#
#   Strategy:
#       1. Train a Random Forest on EACH dataset independently
#          and record feature importances.
#       2. Average the importance scores across both datasets to
#          identify features that are CONSISTENTLY important,
#          not just important for one threat source.
#       3. Apply a threshold (default 1 % mean importance) to
#          select a reduced feature set.
#       4. Re-run within-dataset AND cross-dataset evaluations
#          (all 3 models) on BOTH the full and reduced sets.
#       5. Compare metrics side-by-side and save everything.
#
#   Outputs saved to outputs/:
#       - feature_selection_results.csv
#           Full comparison table (full vs reduced, all models/datasets)
#       - dashboard__feature_selection_within.png
#           Within-dataset comparison dashboard
#       - dashboard__feature_selection_cross.png
#           Cross-dataset comparison dashboard
#       - feature_selection_eval_YYYYMMDD_HHMMSS.txt
#           Full terminal output
#
# Input files (produced by extract_features.py):
#   data/features_phishtank_tranco_dataset.csv
#   data/features_urlhaus_tranco_dataset.csv
#
# Usage:
#   python src/feature_selection_eval.py
#   python src/feature_selection_eval.py --threshold 0.005
#   python src/feature_selection_eval.py --threshold 0.01 --testsize 0.2
# --------------------------------------------------------------

import os        # provides functions for interacting with the operating system, used here for file path construction and directory creation
import sys       # provides access to system-specific parameters and functions, used here to redirect stdout for logging and to exit on errors
import logging   # provides a flexible framework for emitting log messages, used here to log progress and errors with timestamps
import argparse  # provides utilities for parsing command-line arguments, used here to allow --threshold and --testsize flags at runtime
import traceback # provides utilities for extracting and formatting stack traces, used here to log detailed error information on unhandled exceptions
import datetime  # provides classes for manipulating dates and times, used here to generate timestamped log file names

import pandas as pd   # provides high-performance data structures and data analysis tools, used here for loading CSVs, building DataFrames, and saving results
import numpy as np    # provides support for large multi-dimensional arrays and mathematical functions, used here for numerical operations and bar chart positioning
import matplotlib.pyplot as plt              # provides a MATLAB-like plotting interface, used here to create and save dashboard figures
import matplotlib.gridspec as gridspec       # provides a flexible grid layout for subplots, used here to arrange dashboard panels in a structured multi-row layout
import seaborn as sns                        # provides statistical data visualization built on matplotlib, used here to set the dashboard theme

from sklearn.preprocessing  import StandardScaler        # imports the StandardScaler class for feature normalization, used here to scale features for Logistic Regression training only
from sklearn.linear_model   import LogisticRegression    # imports the Logistic Regression classifier, used here as the linear baseline model
from sklearn.ensemble       import RandomForestClassifier # imports the Random Forest classifier, used here as the nonlinear ensemble model and for computing feature importances
from sklearn.tree           import DecisionTreeClassifier # imports the Decision Tree classifier, used here as the interpretable rule-based model
from sklearn.model_selection import train_test_split     # imports the train_test_split function for splitting data into training and testing sets with stratification support
from sklearn.metrics        import (
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

LABEL_CANDIDATES = ["label", "class", "target", "type", "result", "status"]  # defines a list of candidate column names to search for when identifying the label column in a dataset, checked in a case-insensitive manner
LABEL_MAP = {
    "phishing": 1, "phish": 1, "bad": 1, "malicious": 1, "spam": 1, "1": 1,  # maps various string representations of the malicious class to the integer value 1
    "legitimate": 0, "benign": 0, "good": 0, "safe": 0, "0": 0,              # maps various string representations of the legitimate class to the integer value 0
}
METRIC_COLOURS = ["#2E75B6", "#2EA043", "#E67E22", "#8E44AD"]  # defines a list of four hex color codes used consistently for accuracy, precision, recall, and F1-score bars in dashboard charts


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
    os.makedirs(output_dir, exist_ok=True)                                              # creates the output directory if it does not already exist, using exist_ok=True to avoid errors if it already exists
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")                      # generates a timestamp string in YYYYMMDD_HHMMSS format from the current date and time for use in the log file name
    log_path  = os.path.join(output_dir, f"feature_selection_eval_{timestamp}.txt")    # constructs the full path for the timestamped log file by joining the output directory with the formatted filename
    sys.stdout = _Tee(log_path)                                                         # replaces sys.stdout with a _Tee instance so all subsequent print statements and log messages are written to both the terminal and the log file
    logger  = logging.getLogger()                                                       # retrieves the root logger instance to configure global logging behavior for the entire script
    logger.setLevel(logging.INFO)                                                       # sets the minimum logging level to INFO so that all INFO, WARNING, ERROR, and CRITICAL messages are captured and displayed
    handler = logging.StreamHandler(sys.stdout)                                         # creates a StreamHandler that writes log records to sys.stdout, which is now the _Tee object, ensuring logs go to both terminal and file
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))  # sets the log record format to include the time in HH:MM:SS format, the log level in brackets, and the log message
    logger.addHandler(handler)                                                          # attaches the configured handler to the root logger so all log calls use this handler
    logging.getLogger(__name__).info("Results will be saved -> %s", os.path.abspath(log_path))  # logs the absolute path of the log file at INFO level so the user knows exactly where results are being saved
    return log_path  # returns the log file path so the main function can reference it when printing the final confirmation message


log = logging.getLogger(__name__)  # creates a module-level logger named after the current module for use throughout the script, following Python logging best practices


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    # loads a CSV file into a pandas DataFrame, attempting UTF-8 encoding first and falling back to latin-1 if a UnicodeDecodeError is encountered
    try:
        return pd.read_csv(path, low_memory=False)  # attempts to read the CSV file using the default UTF-8 encoding with low_memory=False to avoid dtype inference warnings on large files
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip", low_memory=False)  # if UTF-8 decoding fails, retries with latin-1 encoding and skips any lines that still cause parsing errors


def find_label_column(df: pd.DataFrame) -> str | None:
    # searches the DataFrame column names for a label column by checking each candidate name from LABEL_CANDIDATES in a case-insensitive manner
    lower_map = {c.lower(): c for c in df.columns}  # creates a dictionary mapping each column name lowercased to its original case, enabling case-insensitive column lookup
    for key in LABEL_CANDIDATES:                    # iterates through each candidate label column name defined in LABEL_CANDIDATES
        if key in lower_map:                        # checks whether the current candidate name exists in the lowercase column name mapping
            return lower_map[key]                   # returns the original-case column name corresponding to the matched candidate label name
    return None  # returns None if none of the candidate label column names are found in the DataFrame, signaling that no label column could be identified


def normalize_labels(series: pd.Series) -> pd.Series:
    # normalizes a label series to binary integer values (0 for legitimate, 1 for malicious) regardless of whether the original values are numeric or string-based
    if pd.api.types.is_numeric_dtype(series):           # checks whether the series contains numeric data types, which may already be in the correct binary format
        return pd.to_numeric(series, errors="coerce")   # if the series is numeric, converts it to numeric type and coerces any non-numeric values to NaN for later filtering
    return series.astype(str).str.strip().str.lower().map(LABEL_MAP)  # if the series is non-numeric, converts to string, strips whitespace, converts to lowercase, and maps each value to 0 or 1 using LABEL_MAP


def prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:
    # loads a feature CSV file, identifies and normalizes the label column, removes rows with invalid labels, and returns the feature matrix X and label vector y ready for use
    """Load a feature CSV and return (X, y) using the full dataset."""
    if not os.path.exists(path):                         # checks whether the specified file path exists before attempting to load it
        log.error("Feature file not found: %s", path)   # logs an error message identifying the missing file path if the file does not exist
        sys.exit(1)                                      # exits the program with a non-zero status code to indicate a fatal error caused by the missing input file

    df        = load_dataset(path)       # loads the CSV file at the given path into a DataFrame using the load_dataset helper, which handles encoding fallback automatically
    label_col = find_label_column(df)    # identifies the name of the label column in the loaded DataFrame using the find_label_column helper
    if label_col is None:                # checks whether find_label_column returned None, indicating that no recognizable label column was found in the DataFrame
        log.error("No label column found in %s", path)  # logs an error message identifying the file in which no label column could be found
        sys.exit(1)                      # exits the program with a non-zero status code to indicate a fatal error caused by the missing label column

    y    = normalize_labels(df[label_col])   # normalizes the values in the identified label column to binary integers using the normalize_labels helper
    mask = y.isin([0, 1])                    # creates a boolean mask that is True for rows where the normalized label is either 0 or 1, filtering out any rows with invalid or unrecognized label values
    df   = df[mask].reset_index(drop=True)   # applies the boolean mask to the DataFrame to retain only rows with valid labels, and resets the integer index to be sequential starting from 0
    y    = y[mask].reset_index(drop=True).astype(int)  # applies the same boolean mask to the label series, resets the index, and converts the values to integer type for use in model training and evaluation

    feature_cols = [c for c in df.columns if c not in (label_col, "url")]  # builds a list of feature column names by excluding the label column and the url column, retaining only the engineered numeric features
    X = df[feature_cols].select_dtypes(include=["number"])  # creates the feature matrix X by selecting only the numeric columns from the identified feature columns, ensuring that non-numeric columns are excluded

    log.info(
        "Loaded %s: %d rows, %d features | malicious: %d (%.1f%%)  legitimate: %d (%.1f%%)",
        os.path.basename(path), len(X), len(X.columns),  # logs the base filename, total number of rows, and number of feature columns in the loaded dataset
        int(y.sum()), 100 * y.mean(),                    # logs the count and percentage of malicious samples (label=1) in the dataset
        int((y == 0).sum()), 100 * (1 - y.mean()),       # logs the count and percentage of legitimate samples (label=0) in the dataset
    )
    return X, y  # returns the feature matrix X and label vector y as a tuple for use in feature selection and model evaluation


# --------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------
def select_features(
    X_phish: pd.DataFrame,   # feature matrix for the PhishTank dataset, used to train a Random Forest and compute PhishTank feature importances
    y_phish: pd.Series,      # label vector for the PhishTank dataset, used as the target during Random Forest training for importance computation
    X_url:   pd.DataFrame,   # feature matrix for the URLhaus dataset, used to train a Random Forest and compute URLhaus feature importances
    y_url:   pd.Series,      # label vector for the URLhaus dataset, used as the target during Random Forest training for importance computation
    threshold: float = 0.01, # the minimum mean importance score a feature must have across both datasets to be retained in the reduced feature set; defaults to 0.01 (1%)
) -> tuple[list[str], pd.DataFrame]:
    # trains a Random Forest on each dataset independently, averages the feature importance scores across both datasets, and returns the list of features whose mean importance meets or exceeds the threshold along with the full importance comparison table
    """
    Train a Random Forest on each dataset and average the importance
    scores.  Features whose averaged importance >= threshold are kept.

    Returns
    -------
    selected_features : list[str]
        Column names that survived the threshold cut.
    importance_df : pd.DataFrame
        Full table of importances (both datasets + average).
    """
    log.info("=" * 60)
    log.info("FEATURE SELECTION  (threshold = %.4f)", threshold)  # logs the feature selection threshold being applied so the user can see it in the output
    log.info("=" * 60)

    rf_params = dict(n_estimators=200, random_state=42,
                     n_jobs=-1, class_weight="balanced")  # defines the shared hyperparameter configuration for both Random Forest models used in importance computation: 200 trees, fixed random seed for reproducibility, all CPU cores for parallel training, and balanced class weights to handle any residual class imbalance

    log.info("Fitting Random Forest on PhishTank dataset...")  # logs that the Random Forest for PhishTank feature importance computation is being trained
    rf_phish = RandomForestClassifier(**rf_params)  # instantiates a Random Forest classifier for the PhishTank dataset using the shared hyperparameter configuration defined above
    rf_phish.fit(X_phish, y_phish)                  # trains the Random Forest on the full PhishTank feature matrix and label vector to learn feature importance scores specific to that dataset
    imp_phish = pd.Series(rf_phish.feature_importances_, index=X_phish.columns)  # extracts the feature importance scores from the trained PhishTank Random Forest and stores them in a Series indexed by feature name for easy comparison and lookup

    log.info("Fitting Random Forest on URLhaus dataset...")  # logs that the Random Forest for URLhaus feature importance computation is being trained
    rf_url = RandomForestClassifier(**rf_params)  # instantiates a separate Random Forest classifier for the URLhaus dataset using the same shared hyperparameter configuration to ensure comparability of importance scores
    rf_url.fit(X_url, y_url)                      # trains the Random Forest on the full URLhaus feature matrix and label vector to learn feature importance scores specific to that dataset
    imp_url = pd.Series(rf_url.feature_importances_, index=X_url.columns)  # extracts the feature importance scores from the trained URLhaus Random Forest and stores them in a Series indexed by feature name

    # Build importance comparison table
    importance_df = pd.DataFrame({
        "phishtank":  imp_phish,  # adds the PhishTank importance scores as a column named phishtank in the comparison DataFrame
        "urlhaus":    imp_url,    # adds the URLhaus importance scores as a column named urlhaus in the comparison DataFrame
    })
    importance_df["mean"] = importance_df.mean(axis=1)             # computes the mean importance score for each feature by averaging the PhishTank and URLhaus columns row-wise, representing cross-dataset importance stability
    importance_df = importance_df.sort_values("mean", ascending=False)  # sorts the importance DataFrame in descending order of mean importance so the most consistently important features appear at the top

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE COMPARISON (all 24 features)")
    print("=" * 60)
    print(f"{'Feature':<30s} {'PhishTank':>12s} {'URLhaus':>12s} {'Mean':>12s}  {'Keep?':>6s}")  # prints the header row of the importance comparison table with aligned column labels
    print("-" * 75)  # prints a horizontal separator line below the header row for visual clarity
    for feat, row in importance_df.iterrows():                            # iterates over each row in the sorted importance DataFrame, where feat is the feature name and row contains the importance scores
        keep = "YES" if row["mean"] >= threshold else "no"               # determines whether this feature should be kept by comparing its mean importance score against the threshold, labeling it YES if kept or no if dropped
        print(f"  {feat:<28s} {row['phishtank']:>12.4f} {row['urlhaus']:>12.4f} {row['mean']:>12.4f}  {keep:>6s}")  # prints a formatted row showing the feature name, its importance in each dataset, its mean importance, and whether it is kept or dropped

    selected = importance_df[importance_df["mean"] >= threshold].index.tolist()  # creates the list of selected feature names by filtering the importance DataFrame to retain only features whose mean importance meets or exceeds the threshold

    print(f"\n  Threshold : {threshold:.4f}")                             # prints the threshold value used for feature selection formatted to 4 decimal places
    print(f"  Features kept  : {len(selected)} / {len(importance_df)}")  # prints the count of features that survived the threshold relative to the total number of features evaluated
    print(f"  Features dropped: {len(importance_df) - len(selected)}")   # prints the count of features that were removed because their mean importance fell below the threshold
    dropped = importance_df[importance_df["mean"] < threshold].index.tolist()  # creates a list of dropped feature names by filtering the importance DataFrame to features whose mean importance is below the threshold
    if dropped:
        print(f"  Dropped  : {dropped}")   # prints the list of dropped feature names if any features were removed, providing transparency about which features did not meet the threshold
    print(f"  Selected : {selected}")      # prints the list of selected feature names that will be used in the reduced feature set for Iteration 2 evaluation

    return selected, importance_df  # returns the list of selected feature names and the full importance comparison DataFrame for use in evaluation and dashboard generation


# --------------------------------------------------------------
# Evaluate one configuration (full or reduced feature set)
# --------------------------------------------------------------
def evaluate_models(
    X_train:      pd.DataFrame,  # feature matrix for the training set, either full 24-feature or reduced feature set depending on which configuration is being evaluated
    y_train:      pd.Series,     # label vector for the training set, containing binary integer labels corresponding to the rows in X_train
    X_test:       pd.DataFrame,  # feature matrix for the test set, either full 24-feature or reduced feature set matching the training configuration
    y_test:       pd.Series,     # label vector for the test set, containing binary integer labels corresponding to the rows in X_test
    label:        str,           # a string label identifying the feature set configuration being evaluated, for example Full (24) or Reduced (13), used in result dictionaries
    train_name:   str,           # the name of the training dataset, for example PhishTank or URLhaus, used in result dictionaries and comparison output
    test_name:    str,           # the name of the test dataset, for example URLhaus or PhishTank, used in result dictionaries and comparison output
    results_list: list,          # a shared list to which result dictionaries are appended, allowing results from multiple evaluation calls to be aggregated for saving and reporting
) -> list:
    # trains all three classifiers on the provided training data and evaluates them on the test data, returning a list of result dictionaries and appending them to the shared results list
    """
    Train all 3 models on X_train/y_train and test on X_test/y_test.
    Appends result dicts to results_list and returns them.
    """
    local_results = []  # initializes an empty list to store the result dictionaries for this specific evaluation call, which will be returned at the end of the function

    def _eval(name, model, X_tr, X_te):
        # inner helper function that trains a single model, evaluates it on the test set, computes all metrics, and appends a result dictionary to both local_results and results_list
        model.fit(X_tr, y_train)        # trains the model on the provided training feature matrix using the y_train labels from the outer scope
        y_pred = model.predict(X_te)    # generates predicted class labels for the test feature matrix using the trained model
        cm     = confusion_matrix(y_test, y_pred)  # computes the confusion matrix comparing the true test labels to the predicted labels, producing a 2x2 matrix of TN, FP, FN, TP counts
        r = {
            "feature_set": label,       # stores the feature set label (Full or Reduced with count) to identify which configuration produced this result
            "train_on":    train_name,  # stores the name of the training dataset to identify the source of training data for this result
            "test_on":     test_name,   # stores the name of the test dataset to identify the source of evaluation data for this result
            "model":       name,        # stores the name of the model being evaluated to identify which classifier produced this result
            "accuracy":    round(accuracy_score(y_test, y_pred),  4),                          # computes the overall accuracy as the proportion of correct predictions and rounds to 4 decimal places
            "precision":   round(precision_score(y_test, y_pred, zero_division=0), 4),         # computes precision as true positives divided by predicted positives, with zero_division=0 to handle edge cases, rounded to 4 decimal places
            "recall":      round(recall_score(y_test, y_pred,    zero_division=0), 4),         # computes recall as true positives divided by actual positives, with zero_division=0 to handle edge cases, rounded to 4 decimal places
            "f1_score":    round(f1_score(y_test, y_pred,        zero_division=0), 4),         # computes the F1-score as the harmonic mean of precision and recall, with zero_division=0 to handle edge cases, rounded to 4 decimal places
            "tn": int(cm[0][0]), "fp": int(cm[0][1]),  # extracts the true negative count from position [0][0] and false positive count from position [0][1] of the confusion matrix and converts to integer
            "fn": int(cm[1][0]), "tp": int(cm[1][1]),  # extracts the false negative count from position [1][0] and true positive count from position [1][1] of the confusion matrix and converts to integer
            "n_features":  X_tr.shape[1],  # stores the number of features in the training matrix as the column count of X_tr, used to track whether this is a full or reduced feature set evaluation
        }
        local_results.append(r)   # appends the result dictionary to the local results list for this evaluation call so it can be returned to the caller
        results_list.append(r)    # appends the same result dictionary to the shared results list passed from the main function, allowing aggregation across all evaluation calls
        return model              # returns the trained model object in case the caller needs to access it for feature importance analysis or further inspection

    # Logistic Regression — needs scaling
    scaler  = StandardScaler()           # instantiates a StandardScaler that will normalize features to zero mean and unit variance, which is required for Logistic Regression to converge properly
    X_tr_sc = scaler.fit_transform(X_train)  # fits the scaler on the training data and transforms it simultaneously, learning the mean and standard deviation from training examples only to prevent data leakage
    X_te_sc = scaler.transform(X_test)       # applies the scaler fitted on training data to the test data using transform only, ensuring the same scaling parameters are used without refitting on test data
    _eval("Logistic Regression",
          LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),  # instantiates Logistic Regression with 1000 maximum iterations for convergence, fixed random seed for reproducibility, and balanced class weights
          X_tr_sc, X_te_sc)  # passes the scaled training and test feature matrices to the evaluation helper, as Logistic Regression requires normalized features

    # Random Forest
    _eval("Random Forest",
          RandomForestClassifier(n_estimators=200, random_state=42,
                                 n_jobs=-1, class_weight="balanced"),  # instantiates Random Forest with 200 trees, fixed random seed, all CPU cores for parallel training, and balanced class weights
          X_train, X_test)  # passes the unscaled training and test feature matrices to the evaluation helper, as Random Forest does not require feature normalization

    # Decision Tree
    _eval("Decision Tree (depth=10)",
          DecisionTreeClassifier(max_depth=10, min_samples_split=20,
                                 random_state=42, class_weight="balanced"),  # instantiates Decision Tree with a maximum depth of 10 to prevent overfitting, minimum 20 samples required to split a node, fixed random seed, and balanced class weights
          X_train, X_test)  # passes the unscaled training and test feature matrices to the evaluation helper, as Decision Tree does not require feature normalization

    return local_results  # returns the list of result dictionaries collected during this evaluation call for use by the caller in dashboard generation and comparison output


# --------------------------------------------------------------
# Print a comparison block for one within/cross scenario
# --------------------------------------------------------------
def print_comparison(scenario: str, full_res: list, red_res: list):
    # prints a formatted side-by-side comparison table showing full versus reduced feature set results for a single evaluation scenario, including the delta F1-score for each model
    models = ["Logistic Regression", "Random Forest", "Decision Tree (depth=10)"]  # defines the ordered list of model names to display in the comparison table
    metrics = ["accuracy", "precision", "recall", "f1_score"]  # defines the list of metric keys to display in the comparison table, though only the delta for F1 is shown inline

    print(f"\n{'='*70}")
    print(f"COMPARISON — {scenario}")  # prints the scenario label as the title of this comparison block
    print(f"{'='*70}")
    header = f"  {'Model':<32s} {'Set':>14s} {'Acc':>8s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s}"  # constructs the header row string with aligned column labels for model name, feature set, and the four metrics
    print(header)
    print("  " + "-" * 68)  # prints a separator line below the header row

    full_map = {r["model"]: r for r in full_res}  # creates a dictionary mapping model names to their full feature set result dictionaries for quick lookup during the comparison loop
    red_map  = {r["model"]: r for r in red_res}   # creates a dictionary mapping model names to their reduced feature set result dictionaries for quick lookup during the comparison loop

    for m in models:                    # iterates over each model name in the defined order to print the comparison rows
        fr = full_map.get(m)            # retrieves the full feature set result dictionary for the current model, or None if not found
        rr = red_map.get(m)             # retrieves the reduced feature set result dictionary for the current model, or None if not found
        if fr:
            print(f"  {m:<32s} {fr['feature_set']:>14s} "
                  f"{fr['accuracy']:>8.4f} {fr['precision']:>8.4f} "
                  f"{fr['recall']:>8.4f} {fr['f1_score']:>8.4f}")  # prints the full feature set row for this model showing all four metrics formatted to 4 decimal places
        if rr:
            delta_f1 = rr["f1_score"] - (fr["f1_score"] if fr else 0)  # computes the F1-score delta as the difference between the reduced and full feature set F1-scores for this model
            sign = "+" if delta_f1 >= 0 else ""                         # prepends a plus sign to positive or zero deltas for display clarity, leaving negative deltas with their natural minus sign
            print(f"  {m:<32s} {rr['feature_set']:>14s} "
                  f"{rr['accuracy']:>8.4f} {rr['precision']:>8.4f} "
                  f"{rr['recall']:>8.4f} {rr['f1_score']:>8.4f}"
                  f"  (ΔF1={sign}{delta_f1:.4f})")  # prints the reduced feature set row for this model with all four metrics and the delta F1-score annotation
        print()  # prints a blank line after each model's rows to visually separate models in the comparison table


# --------------------------------------------------------------
# Dashboard helpers
# --------------------------------------------------------------
def _bar_comparison_ax(
    ax, full_results: list, red_results: list,
    title: str, highlight_recall: bool = False
):
    # renders a grouped bar chart on the provided axes comparing full versus reduced feature set F1-scores for all three models, with value labels on each bar
    """
    Grouped bar chart: for each model show full vs reduced F1
    (and optionally highlight recall change).
    """
    models  = [r["model"].replace(" (depth=10)", "\n(depth=10)")
               for r in full_results]  # extracts model names from the full results list and replaces the depth notation with a newline for better readability on the x-axis
    x       = np.arange(len(models))   # creates an array of evenly spaced x positions, one per model, used as the center positions for the grouped bars
    width   = 0.28  # defines the width of each bar so that the two bars per model fit side by side without overlapping

    full_f1 = [r["f1_score"] for r in full_results]  # extracts the F1-scores for the full feature set from the full results list, used as bar heights for the first group
    red_f1  = [r["f1_score"] for r in red_results]   # extracts the F1-scores for the reduced feature set from the reduced results list, used as bar heights for the second group
    full_re = [r["recall"]   for r in full_results]  # extracts the recall values for the full feature set, available for potential use in highlighting recall changes
    red_re  = [r["recall"]   for r in red_results]   # extracts the recall values for the reduced feature set, available for potential use in highlighting recall changes

    b1 = ax.bar(x - width/2, full_f1, width,
                label="Full features — F1", color="#2E75B6", alpha=0.88)  # draws the full feature set F1 bars positioned to the left of each x position using blue color with slight transparency
    b2 = ax.bar(x + width/2, red_f1,  width,
                label="Reduced features — F1", color="#E67E22", alpha=0.88)  # draws the reduced feature set F1 bars positioned to the right of each x position using orange color with slight transparency

    for bar, val in zip(b1, full_f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7, color="white")  # adds a value label centered above each full feature set bar showing the F1-score to 4 decimal places in white text
    for bar, val in zip(b2, red_f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7, color="white")  # adds a value label centered above each reduced feature set bar showing the F1-score to 4 decimal places in white text

    ax.set_xticks(x)                                              # sets the x-axis tick positions to align with the center of each model's bar group
    ax.set_xticklabels(models, color="white", fontsize=8)         # sets the x-axis tick labels to the model names in white text at size 8 for readability against the dark background
    ax.set_ylim(min(min(full_f1), min(red_f1)) - 0.04, 1.08)     # sets the y-axis limits with a lower bound slightly below the minimum F1-score and an upper bound above 1.0 to provide space for value labels
    ax.set_ylabel("F1-Score", color="white", fontsize=9)          # labels the y-axis as F1-Score in white text for contrast against the dark background
    ax.set_title(title, color="white", fontweight="bold", fontsize=9)  # sets the subplot title in bold white text using the title string passed by the caller
    ax.tick_params(colors="white")                                # sets all tick mark and tick label colors to white for visibility against the dark background
    ax.spines[:].set_color("#444")                                # sets the color of all axis spine borders to dark gray for a subtle framing effect against the dark background
    ax.legend(loc="lower right", fontsize=7, facecolor="#21262D", labelcolor="white")  # adds a legend in the lower right corner with dark background and white label text to identify the full versus reduced bars


def _delta_ax(ax, full_results: list, red_results: list, metric: str, title: str):
    # renders a horizontal bar chart on the provided axes showing the change in a specified metric between the reduced and full feature sets for each model, with green bars for improvements and red bars for degradations
    """Horizontal bar chart showing Δmetric (reduced − full)."""
    models  = [r["model"].replace(" (depth=10)", "\n(depth=10)")
               for r in full_results]  # extracts model names and formats the depth notation with a newline for better readability on the y-axis labels
    full_m  = {r["model"]: r[metric] for r in full_results}  # creates a dictionary mapping each model name to its full feature set value for the specified metric
    red_m   = {r["model"]: r[metric] for r in red_results}   # creates a dictionary mapping each model name to its reduced feature set value for the specified metric
    deltas  = [red_m[r["model"]] - full_m[r["model"]] for r in full_results]  # computes the delta for each model as the reduced metric value minus the full metric value, so positive values indicate improvement
    colours = ["#2EA043" if d >= 0 else "#C0392B" for d in deltas]  # assigns green color to non-negative deltas indicating improvement or no change, and red color to negative deltas indicating degradation

    y = np.arange(len(models))  # creates an array of y positions, one per model, used to position the horizontal bars
    bars = ax.barh(y, deltas, color=colours, alpha=0.85)  # draws horizontal bars at each y position with heights equal to the delta values and colors based on direction of change
    for bar, val in zip(bars, deltas):
        sign = "+" if val >= 0 else ""  # prepends a plus sign to non-negative delta values for display, leaving negative values with their natural minus sign
        ax.text(bar.get_width() + (0.0002 if val >= 0 else -0.0002),
                bar.get_y() + bar.get_height()/2,
                f"{sign}{val:.4f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8, color="white")  # places a delta value label at the end of each bar, positioning it to the right of positive bars and to the left of negative bars, formatted to 4 decimal places
    ax.set_yticks(y)                                                # sets the y-axis tick positions to align with each horizontal bar
    ax.set_yticklabels(models, color="white", fontsize=8)           # sets the y-axis tick labels to model names in white text at size 8
    ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")      # draws a vertical dashed reference line at x=0 to visually separate positive from negative deltas
    ax.set_xlabel(f"Δ {metric.replace('_', ' ').title()} (reduced − full)",
                  color="white", fontsize=9)  # labels the x-axis with the metric name formatted in title case, preceded by a delta symbol, in white text
    ax.set_title(title, color="white", fontweight="bold", fontsize=9)  # sets the subplot title in bold white text using the title string passed by the caller
    ax.tick_params(colors="white")      # sets all tick mark and label colors to white for visibility against the dark background
    ax.spines[:].set_color("#444")      # sets the color of all axis spine borders to dark gray for subtle framing


def _metrics_table_ax(ax, results: list, title: str):
    # renders a compact metrics table on the provided axes showing model name, accuracy, precision, recall, F1-score, and feature count for each result in the provided list
    """Compact metrics table in a subplot."""
    ax.axis("off")                      # turns off the axis lines and ticks since this subplot displays a table rather than a traditional chart
    ax.set_facecolor("#161B22")         # sets the subplot background color to a dark shade for visual consistency with the rest of the dashboard

    table_data = [
        [r["model"],
         f"{r['accuracy']:.4f}",    # formats the accuracy value to 4 decimal places for display in the table cell
         f"{r['precision']:.4f}",   # formats the precision value to 4 decimal places for display in the table cell
         f"{r['recall']:.4f}",      # formats the recall value to 4 decimal places for display in the table cell
         f"{r['f1_score']:.4f}",    # formats the F1-score value to 4 decimal places for display in the table cell
         str(r["n_features"])]      # converts the feature count to a string for display in the table cell
        for r in results  # iterates over each result dictionary in the provided list to build one row per model
    ]
    col_labels = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Feats"]  # defines the column header labels for the metrics table
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="center", loc="center")  # creates the matplotlib table object with the data rows, column headers, center-aligned cell text, and centered position within the axes
    tbl.auto_set_font_size(False)  # disables automatic font size adjustment to allow manual control over the font size
    tbl.set_fontsize(8)            # sets the font size for all table cells to 8 points for compact readability within the dashboard layout
    tbl.scale(1, 1.9)              # scales the table with no horizontal change and 1.9x vertical scaling to increase row height for better readability

    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#C0392B")                            # sets the background color of each header cell (row index 0) to red for visual emphasis and distinction from data rows
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")    # sets the header cell text to white and bold for readability against the red background

    row_colours = ["#1C2128", "#21262D"]  # defines two alternating dark background colors for data rows to create a striped effect that improves row readability
    for i in range(len(table_data)):      # iterates over each data row index to apply alternating background colors
        for j in range(len(col_labels)):  # iterates over each column index to apply the background color to every cell in the current row
            tbl[(i+1, j)].set_facecolor(row_colours[i % 2])    # sets the cell background color by alternating between the two row colors based on whether the row index is even or odd
            tbl[(i+1, j)].set_text_props(color="white")         # sets the data cell text color to white for readability against the dark background

    ax.set_title(title, color="white", fontsize=8, fontweight="bold", pad=6)  # sets the table subplot title in bold white text at size 8 with 6 points of padding between the title and the table


# --------------------------------------------------------------
# Full dashboard: within-dataset or cross-dataset comparison
# --------------------------------------------------------------
def save_comparison_dashboard(
    full_within: dict,     # dictionary mapping dataset names to lists of result dictionaries for the full 24-feature within-dataset evaluation
    red_within:  dict,     # dictionary mapping dataset names to lists of result dictionaries for the reduced feature within-dataset evaluation
    full_cross:  dict,     # dictionary mapping (train_name, test_name) tuples to lists of result dictionaries for the full 24-feature cross-dataset evaluation
    red_cross:   dict,     # dictionary mapping (train_name, test_name) tuples to lists of result dictionaries for the reduced feature cross-dataset evaluation
    importance_df: pd.DataFrame,  # the full feature importance comparison DataFrame containing PhishTank, URLhaus, and mean importance scores for all 24 features
    selected:    list[str],        # the list of feature names that survived the threshold cut and are used in the reduced feature set
    output_dir:  str,              # the directory path where the dashboard PNG files will be saved
    threshold:   float,            # the importance threshold used for feature selection, displayed in the dashboard title for reference
):
    # generates and saves two dashboard PNG files: one comparing full versus reduced feature set performance on within-dataset evaluation, and one comparing performance on cross-dataset evaluation
    """
    Two separate dashboards:
      1. Within-dataset comparison
      2. Cross-dataset comparison
    """
    sns.set_theme(style="whitegrid", font_scale=0.85)  # sets the seaborn visualization theme to whitegrid with a slightly reduced font scale for compact dashboard layout

    # ── Dashboard 1: Within-dataset ──────────────────────────────
    fig1 = plt.figure(figsize=(22, 15))         # creates the first dashboard figure with dimensions of 22 inches wide by 15 inches tall to provide ample space for all subplots
    fig1.patch.set_facecolor("#0D1117")         # sets the overall figure background color to a very dark shade for the dashboard theme
    fig1.suptitle(
        f"Iteration 2 — Feature Selection Impact: Within-Dataset Evaluation\n"
        f"Full (24 features) vs Reduced ({len(selected)} features, threshold={threshold:.3f})",
        fontsize=13, fontweight="bold", color="white", y=0.98  # sets the overall figure title with the iteration label, feature counts, and threshold, positioned near the top with bold white text
    )
    gs1 = gridspec.GridSpec(3, 4, figure=fig1,
                            hspace=0.60, wspace=0.45,
                            left=0.06, right=0.97, top=0.91, bottom=0.05)  # creates a 3-row by 4-column grid layout for the within-dataset dashboard with specified spacing and margins

    # Row 0: Feature importance heatmap (averaged)
    ax_imp = fig1.add_subplot(gs1[0, :])    # creates a subplot spanning all 4 columns of the first row for the feature importance bar chart
    ax_imp.set_facecolor("#161B22")         # sets the feature importance subplot background to dark for dashboard consistency
    top_imp = importance_df.head(24)        # selects all 24 features from the sorted importance DataFrame for display in the importance chart
    x_pos   = np.arange(len(top_imp))      # creates an array of x positions for the grouped importance bars, one position per feature
    width   = 0.28                          # defines the bar width for the grouped importance chart
    bars_p  = ax_imp.bar(x_pos - width/2, top_imp["phishtank"],
                         width, color="#2E75B6", alpha=0.85, label="PhishTank")  # draws the PhishTank importance bars in blue, positioned to the left of each feature's x position
    bars_u  = ax_imp.bar(x_pos + width/2, top_imp["urlhaus"],
                         width, color="#E67E22", alpha=0.85, label="URLhaus")    # draws the URLhaus importance bars in orange, positioned to the right of each feature's x position
    # Draw threshold line
    ax_imp.axhline(threshold, color="#C0392B", linewidth=1.5, linestyle="--",
                   label=f"Threshold ({threshold:.3f})")  # draws a horizontal dashed red line at the threshold value to visually indicate the selection cutoff across all features
    ax_imp.set_xticks(x_pos)
    ax_imp.set_xticklabels(top_imp.index, rotation=40, ha="right",
                           color="white", fontsize=8)  # sets the x-axis tick labels to feature names rotated 40 degrees for readability, in white text at size 8
    ax_imp.set_ylabel("Importance Score", color="white", fontsize=9)  # labels the y-axis as Importance Score in white text
    ax_imp.set_title(
        "Feature Importances: PhishTank vs URLhaus (averaged for selection)",
        color="white", fontweight="bold", fontsize=10  # sets the importance chart title in bold white text
    )
    ax_imp.tick_params(colors="white")      # sets tick mark and label colors to white
    ax_imp.spines[:].set_color("#444")      # sets spine border colors to dark gray
    ax_imp.legend(loc="upper right", fontsize=8,
                  facecolor="#21262D", labelcolor="white")  # adds a legend identifying PhishTank and URLhaus bars and the threshold line

    # Shade dropped features in red
    for i, feat in enumerate(top_imp.index):
        if feat not in selected:
            ax_imp.axvspan(i - 0.5, i + 0.5, color="#C0392B", alpha=0.10)  # adds a light red vertical shading band behind each feature that was dropped, visually highlighting which features did not meet the threshold

    # Row 1: Per-dataset F1 comparison bars + delta
    dataset_keys = list(full_within.keys())  # retrieves the dataset name keys from the full within-dataset results dictionary to iterate over for bar chart generation
    for col_idx, ds_name in enumerate(dataset_keys[:2]):  # iterates over up to 2 dataset names, one for each half of the second dashboard row
        ds_label  = ds_name.replace("features_", "").replace("_", " ").title()  # formats the dataset name for display by removing the features_ prefix, replacing underscores with spaces, and converting to title case
        ax_bar    = fig1.add_subplot(gs1[1, col_idx * 2 : col_idx * 2 + 2])     # creates a subplot spanning 2 columns in the second row for this dataset's F1 comparison bar chart
        ax_bar.set_facecolor("#161B22")  # sets the bar chart subplot background to dark
        _bar_comparison_ax(ax_bar,
                           full_within[ds_name],
                           red_within[ds_name],
                           f"F1 Comparison — {ds_label}")  # calls the bar comparison helper to render the full versus reduced F1 comparison bars for this dataset

    # Row 2: Metrics tables
    for col_idx, ds_name in enumerate(dataset_keys[:2]):  # iterates over up to 2 dataset names to create paired full and reduced metrics tables in the third dashboard row
        ds_label = ds_name.replace("features_", "").replace("_", " ").title()  # formats the dataset name for display in table titles

        ax_full = fig1.add_subplot(gs1[2, col_idx * 2])      # creates a subplot in the left column of the third row for this dataset's full feature set metrics table
        ax_full.set_facecolor("#161B22")                      # sets the table subplot background to dark
        _metrics_table_ax(ax_full, full_within[ds_name],
                          f"{ds_label} — Full (24 features)")  # calls the metrics table helper to render the full feature set metrics table for this dataset

        ax_red = fig1.add_subplot(gs1[2, col_idx * 2 + 1])   # creates a subplot in the right column of the third row for this dataset's reduced feature set metrics table
        ax_red.set_facecolor("#161B22")                       # sets the table subplot background to dark
        _metrics_table_ax(ax_red, red_within[ds_name],
                          f"{ds_label} — Reduced ({len(selected)} features)")  # calls the metrics table helper to render the reduced feature set metrics table for this dataset

    save_path1 = os.path.join(output_dir, "dashboard__feature_selection_within.png")  # constructs the output file path for the within-dataset dashboard PNG
    plt.savefig(save_path1, dpi=150, bbox_inches="tight",
                facecolor=fig1.get_facecolor())  # saves the within-dataset dashboard figure as a PNG at 150 DPI with tight bounding box and the dark figure background color
    log.info("Dashboard saved -> %s", os.path.abspath(save_path1))  # logs the absolute path of the saved within-dataset dashboard for user reference
    plt.close(fig1)  # closes the within-dataset dashboard figure to release memory before creating the cross-dataset dashboard

    # ── Dashboard 2: Cross-dataset ───────────────────────────────
    fig2 = plt.figure(figsize=(22, 16))   # creates the second dashboard figure for cross-dataset comparison with dimensions of 22 by 16 inches
    fig2.patch.set_facecolor("#0D1117")   # sets the cross-dataset dashboard background to the same dark shade as the within-dataset dashboard
    fig2.suptitle(
        f"Iteration 2 — Feature Selection Impact: Cross-Dataset Evaluation\n"
        f"Full (24 features) vs Reduced ({len(selected)} features, threshold={threshold:.3f})",
        fontsize=13, fontweight="bold", color="white", y=0.98  # sets the cross-dataset dashboard title with iteration label, feature counts, and threshold
    )
    gs2 = gridspec.GridSpec(3, 4, figure=fig2,
                            hspace=0.60, wspace=0.45,
                            left=0.06, right=0.97, top=0.91, bottom=0.05)  # creates a 3-row by 4-column grid layout for the cross-dataset dashboard with the same spacing and margins as the within-dataset dashboard

    cross_keys = list(full_cross.keys())  # retrieves the (train_name, test_name) tuple keys from the full cross-dataset results dictionary to iterate over for chart generation

    # Row 0: F1 comparison bars (both cross directions)
    for col_idx, key in enumerate(cross_keys[:2]):  # iterates over up to 2 cross-evaluation directions, one for each half of the first cross-dataset dashboard row
        train_name, test_name = key  # unpacks the (train_name, test_name) tuple key into individual variables for use in chart titles
        ax_bar = fig2.add_subplot(gs2[0, col_idx * 2 : col_idx * 2 + 2])  # creates a subplot spanning 2 columns in the first row for this direction's F1 comparison bar chart
        ax_bar.set_facecolor("#161B22")  # sets the bar chart subplot background to dark
        _bar_comparison_ax(ax_bar,
                           full_cross[key],
                           red_cross[key],
                           f"F1: Train {train_name} → Test {test_name}",
                           highlight_recall=True)  # calls the bar comparison helper to render the full versus reduced F1 comparison for this cross-evaluation direction

    # Row 1: Recall delta charts (most important for cross-eval)
    for col_idx, key in enumerate(cross_keys[:2]):  # iterates over up to 2 cross-evaluation directions to create recall delta charts in the second cross-dataset dashboard row
        train_name, test_name = key  # unpacks the direction tuple key
        ax_delta = fig2.add_subplot(gs2[1, col_idx * 2 : col_idx * 2 + 2])  # creates a subplot spanning 2 columns in the second row for this direction's recall delta chart
        ax_delta.set_facecolor("#161B22")  # sets the delta chart subplot background to dark
        _delta_ax(ax_delta,
                  full_cross[key],
                  red_cross[key],
                  "recall",
                  f"ΔRecall: Train {train_name} → Test {test_name}")  # calls the delta chart helper to render the recall change between full and reduced feature sets for this cross-evaluation direction

    # Row 2: Metrics tables
    for col_idx, key in enumerate(cross_keys[:2]):  # iterates over up to 2 cross-evaluation directions to create paired full and reduced metrics tables in the third cross-dataset dashboard row
        train_name, test_name = key  # unpacks the direction tuple key for use in table titles

        ax_full = fig2.add_subplot(gs2[2, col_idx * 2])  # creates a subplot in the left column of the third row for this direction's full feature set metrics table
        ax_full.set_facecolor("#161B22")  # sets the table subplot background to dark
        _metrics_table_ax(ax_full, full_cross[key],
                          f"Full — Train {train_name} → Test {test_name}")  # calls the metrics table helper to render the full feature set cross-evaluation metrics table

        ax_red = fig2.add_subplot(gs2[2, col_idx * 2 + 1])  # creates a subplot in the right column of the third row for this direction's reduced feature set metrics table
        ax_red.set_facecolor("#161B22")  # sets the table subplot background to dark
        _metrics_table_ax(ax_red, red_cross[key],
                          f"Reduced — Train {train_name} → Test {test_name}")  # calls the metrics table helper to render the reduced feature set cross-evaluation metrics table

    save_path2 = os.path.join(output_dir, "dashboard__feature_selection_cross.png")  # constructs the output file path for the cross-dataset dashboard PNG
    plt.savefig(save_path2, dpi=150, bbox_inches="tight",
                facecolor=fig2.get_facecolor())  # saves the cross-dataset dashboard figure as a PNG at 150 DPI with tight bounding box and the dark figure background color
    log.info("Dashboard saved -> %s", os.path.abspath(save_path2))  # logs the absolute path of the saved cross-dataset dashboard for user reference
    plt.close(fig2)  # closes the cross-dataset dashboard figure to release memory after saving


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main(threshold: float = 0.01, test_size: float = 0.2) -> None:
    # orchestrates the full Iteration 2 feature selection evaluation: loads datasets, selects features, runs within-dataset and cross-dataset evaluations for both full and reduced feature sets, saves results, and generates dashboards
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # creates the outputs directory if it does not already exist, ensuring results can be saved without errors
    log_path    = setup_logging(OUTPUT_DIR)  # configures logging and stdout redirection to the timestamped log file, returning the log file path for the final confirmation message
    all_results = []  # initializes an empty list that will accumulate all result dictionaries from every model, dataset, feature set, and evaluation direction throughout the script

    # ----------------------------------------------------------
    # 1. Load datasets
    # ----------------------------------------------------------
    log.info("Loading datasets...")  # logs that the dataset loading phase is beginning
    X_phish, y_phish = prepare(PHISHTANK_FILE)  # loads the PhishTank feature dataset and returns the feature matrix X_phish and label vector y_phish using the prepare helper
    X_url,   y_url   = prepare(URLHAUS_FILE)    # loads the URLhaus feature dataset and returns the feature matrix X_url and label vector y_url using the prepare helper

    # Align columns (both datasets share the same 24 features)
    shared_cols = sorted(set(X_phish.columns) & set(X_url.columns))  # computes the intersection of column names from both datasets and sorts them alphabetically to ensure a consistent feature ordering for cross-dataset evaluation
    X_phish = X_phish[shared_cols]  # reindexes the PhishTank feature matrix to use only the shared columns in the sorted order, ensuring column alignment with the URLhaus dataset
    X_url   = X_url[shared_cols]    # reindexes the URLhaus feature matrix to use only the shared columns in the same sorted order, ensuring column alignment with the PhishTank dataset

    # ----------------------------------------------------------
    # 2. Feature selection using full datasets
    # ----------------------------------------------------------
    selected, importance_df = select_features(
        X_phish, y_phish, X_url, y_url, threshold=threshold
    )  # calls the feature selection function with both full datasets and the specified threshold, returning the list of selected feature names and the full importance comparison DataFrame

    if len(selected) == 0:
        log.error("No features survived the threshold! Lower --threshold.")  # logs an error if no features survived the threshold, which would make further evaluation impossible
        sys.exit(1)  # exits with a non-zero status code if no features were selected, preventing the script from proceeding with an empty feature set

    # ----------------------------------------------------------
    # 3. Within-dataset evaluation (full vs reduced)
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("WITHIN-DATASET EVALUATION")
    print(f"{'='*60}")

    full_within = {}  # initializes an empty dictionary to store full feature set within-dataset results, keyed by dataset name
    red_within  = {}  # initializes an empty dictionary to store reduced feature set within-dataset results, keyed by dataset name

    for ds_name, X_full, y_full in [
        ("features_phishtank_tranco_dataset", X_phish, y_phish),  # defines the PhishTank dataset entry with its name, feature matrix, and label vector
        ("features_urlhaus_tranco_dataset",   X_url,   y_url),    # defines the URLhaus dataset entry with its name, feature matrix, and label vector
    ]:
        label = ds_name.replace("features_", "").replace("_tranco_dataset", "").title()  # formats the dataset name for display by removing prefixes and suffixes, replacing underscores with spaces, and converting to title case
        X_tr_f, X_te_f, y_tr, y_te = train_test_split(
            X_full, y_full,
            test_size=test_size, random_state=42, stratify=y_full
        )  # splits the full feature dataset into 80/20 training and test sets using stratification to preserve the class distribution, with a fixed random seed for reproducibility

        # -- Full feature set
        print(f"\n--- {label}  |  Full ({len(shared_cols)} features) ---")  # prints a section header identifying the current dataset and feature set configuration being evaluated
        f_res = evaluate_models(
            X_tr_f, y_tr, X_te_f, y_te,
            label=f"Full ({len(shared_cols)})",
            train_name=label, test_name=label,
            results_list=all_results,
        )  # calls evaluate_models to train and evaluate all three classifiers on the full 24-feature training and test sets for this dataset, appending results to all_results
        full_within[ds_name] = f_res  # stores the full feature set results for this dataset in the full_within dictionary for use in dashboard generation
        for r in f_res:
            print(f"  {r['model']:<35s}  Acc={r['accuracy']:.4f}  "
                  f"Prec={r['precision']:.4f}  Rec={r['recall']:.4f}  "
                  f"F1={r['f1_score']:.4f}")  # prints a one-line summary of each model's performance metrics formatted to 4 decimal places

        # -- Reduced feature set
        X_tr_r = X_tr_f[selected]  # creates the reduced training feature matrix by selecting only the columns whose names appear in the selected features list
        X_te_r = X_te_f[selected]  # creates the reduced test feature matrix by selecting only the same columns as the reduced training matrix, maintaining consistency
        print(f"\n--- {label}  |  Reduced ({len(selected)} features) ---")  # prints a section header identifying the current dataset and reduced feature set configuration
        r_res = evaluate_models(
            X_tr_r, y_tr, X_te_r, y_te,
            label=f"Reduced ({len(selected)})",
            train_name=label, test_name=label,
            results_list=all_results,
        )  # calls evaluate_models to train and evaluate all three classifiers on the reduced feature training and test sets for this dataset, appending results to all_results
        red_within[ds_name] = r_res  # stores the reduced feature set results for this dataset in the red_within dictionary for use in dashboard generation
        for r in r_res:
            print(f"  {r['model']:<35s}  Acc={r['accuracy']:.4f}  "
                  f"Prec={r['precision']:.4f}  Rec={r['recall']:.4f}  "
                  f"F1={r['f1_score']:.4f}")  # prints a one-line summary of each model's reduced feature set performance metrics

        print_comparison(f"Within-dataset — {label}",
                         full_within[ds_name],
                         red_within[ds_name])  # calls the comparison printer to display a side-by-side table of full versus reduced results with delta F1 annotations for this dataset

    # ----------------------------------------------------------
    # 4. Cross-dataset evaluation (full vs reduced)
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("CROSS-DATASET EVALUATION")
    print(f"{'='*60}")

    full_cross = {}  # initializes an empty dictionary to store full feature set cross-dataset results, keyed by (train_name, test_name) tuples
    red_cross  = {}  # initializes an empty dictionary to store reduced feature set cross-dataset results, keyed by (train_name, test_name) tuples

    for train_name, test_name, X_tr_full, y_tr, X_te_full, y_te in [
        ("PhishTank", "URLhaus",   X_phish, y_phish, X_url,   y_url),    # defines Direction 1: train on full PhishTank dataset, test on full URLhaus dataset
        ("URLhaus",   "PhishTank", X_url,   y_url,   X_phish, y_phish),  # defines Direction 2: train on full URLhaus dataset, test on full PhishTank dataset
    ]:
        key = (train_name, test_name)  # creates the dictionary key as a tuple of the training and test dataset names for storing results

        # Full
        print(f"\n--- Cross: Train {train_name} → Test {test_name}"
              f"  |  Full ({len(shared_cols)} features) ---")  # prints a section header identifying the cross-evaluation direction and feature set configuration
        f_res = evaluate_models(
            X_tr_full, y_tr, X_te_full, y_te,
            label=f"Full ({len(shared_cols)})",
            train_name=train_name, test_name=test_name,
            results_list=all_results,
        )  # calls evaluate_models to train on the full training dataset and evaluate on the full test dataset using all 24 features, appending results to all_results
        full_cross[key] = f_res  # stores the full feature set cross-evaluation results for this direction in the full_cross dictionary
        for r in f_res:
            print(f"  {r['model']:<35s}  Acc={r['accuracy']:.4f}  "
                  f"Prec={r['precision']:.4f}  Rec={r['recall']:.4f}  "
                  f"F1={r['f1_score']:.4f}")  # prints a one-line summary of each model's full feature cross-evaluation metrics

        # Reduced
        X_tr_r = X_tr_full[selected]  # creates the reduced training feature matrix for cross-evaluation by selecting only the survived feature columns from the full training dataset
        X_te_r = X_te_full[selected]  # creates the reduced test feature matrix for cross-evaluation by selecting the same survived feature columns from the full test dataset
        print(f"\n--- Cross: Train {train_name} → Test {test_name}"
              f"  |  Reduced ({len(selected)} features) ---")  # prints a section header for the reduced feature set cross-evaluation
        r_res = evaluate_models(
            X_tr_r, y_tr, X_te_r, y_te,
            label=f"Reduced ({len(selected)})",
            train_name=train_name, test_name=test_name,
            results_list=all_results,
        )  # calls evaluate_models to train on the reduced training dataset and evaluate on the reduced test dataset, appending results to all_results
        red_cross[key] = r_res  # stores the reduced feature set cross-evaluation results for this direction in the red_cross dictionary
        for r in r_res:
            print(f"  {r['model']:<35s}  Acc={r['accuracy']:.4f}  "
                  f"Prec={r['precision']:.4f}  Rec={r['recall']:.4f}  "
                  f"F1={r['f1_score']:.4f}")  # prints a one-line summary of each model's reduced feature cross-evaluation metrics

        print_comparison(f"Cross-dataset — Train {train_name} → Test {test_name}",
                         full_cross[key], red_cross[key])  # calls the comparison printer to display a side-by-side table of full versus reduced cross-evaluation results with delta F1 annotations

    # ----------------------------------------------------------
    # 5. Save results CSV
    # ----------------------------------------------------------
    results_df   = pd.DataFrame(all_results)  # converts the accumulated list of all result dictionaries into a pandas DataFrame for structured saving and reporting
    results_path = os.path.join(OUTPUT_DIR, "feature_selection_results.csv")  # constructs the output file path for the results CSV
    results_df.to_csv(results_path, index=False)  # saves the results DataFrame to a CSV file without the DataFrame index column for clean output
    log.info("Results saved -> %s", os.path.abspath(results_path))  # logs the absolute path of the saved results CSV for user reference

    # Save importance table
    imp_path = os.path.join(OUTPUT_DIR, "feature_importances_comparison.csv")  # constructs the output file path for the feature importances comparison CSV
    importance_df.to_csv(imp_path)  # saves the feature importance comparison DataFrame to a CSV file, including the index which contains the feature names
    log.info("Importance table saved -> %s", os.path.abspath(imp_path))  # logs the absolute path of the saved importance table CSV for user reference

    # ----------------------------------------------------------
    # 6. Print full summary
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print("FULL RESULTS SUMMARY — FEATURE SELECTION ITERATION")
    print(f"{'='*70}")
    print(results_df[[
        "feature_set", "train_on", "test_on", "model",
        "accuracy", "precision", "recall", "f1_score", "n_features"
    ]].to_string(index=False))  # prints the selected columns of the full results DataFrame as a formatted string without the index for a clean console summary

    # Analysis notes
    print(f"\n{'='*70}")
    print("ANALYSIS — KEY FINDINGS")
    print(f"{'='*70}")
    print(f"\n  Selected {len(selected)} features (threshold={threshold:.4f}):")  # prints the count of selected features and the threshold applied
    for f in selected:
        imp_mean = importance_df.loc[f, "mean"]       # retrieves the mean importance score for this feature from the importance DataFrame
        imp_pt   = importance_df.loc[f, "phishtank"]  # retrieves the PhishTank importance score for this feature
        imp_url  = importance_df.loc[f, "urlhaus"]    # retrieves the URLhaus importance score for this feature
        print(f"    {f:<30s}  mean={imp_mean:.4f}  "
              f"phishtank={imp_pt:.4f}  urlhaus={imp_url:.4f}")  # prints each selected feature with its mean, PhishTank, and URLhaus importance scores formatted to 4 decimal places

    # Cross-eval recall delta summary
    print("\n  Cross-dataset recall change (reduced − full):")  # prints the header for the cross-dataset recall change summary
    for key in full_cross:
        train_n, test_n = key  # unpacks the direction tuple key into individual train and test dataset names
        fm = {r["model"]: r for r in full_cross[key]}   # creates a dictionary mapping model names to their full feature set cross-evaluation results for this direction
        rm = {r["model"]: r for r in red_cross[key]}    # creates a dictionary mapping model names to their reduced feature set cross-evaluation results for this direction
        for m in ["Logistic Regression", "Random Forest", "Decision Tree (depth=10)"]:
            delta = rm[m]["recall"] - fm[m]["recall"]  # computes the recall change as the reduced feature set recall minus the full feature set recall for this model and direction
            sign  = "+" if delta >= 0 else ""           # prepends a plus sign to non-negative deltas for display clarity
            print(f"    Train {train_n:>10s} → Test {test_n:<12s}  "
                  f"{m:<35s}  ΔRecall={sign}{delta:.4f}")  # prints the recall delta for each model in each cross-evaluation direction

    # ----------------------------------------------------------
    # 7. Dashboards
    # ----------------------------------------------------------
    save_comparison_dashboard(
        full_within, red_within,
        full_cross,  red_cross,
        importance_df, selected,
        OUTPUT_DIR, threshold,
    )  # calls the dashboard generation function with all accumulated results to create and save the within-dataset and cross-dataset comparison dashboard PNG files

    print("\nFeature selection evaluation complete.")  # prints a completion message to confirm that all evaluations, results saving, and dashboard generation have finished successfully

    if isinstance(sys.stdout, _Tee):
        sys.stdout.close()          # closes the _Tee log file if stdout is still redirected, ensuring all buffered output is written to disk before restoring normal stdout
        sys.stdout = sys.__stdout__ # restores sys.stdout to the original terminal stdout so any subsequent print statements go directly to the terminal rather than the log file
    print(f"\nFull results saved -> {os.path.abspath(log_path)}")  # prints the absolute path of the log file as a final confirmation after stdout has been restored to normal terminal output


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------
def parse_args():
    # parses command-line arguments to allow the user to customize the feature selection threshold and test set size without modifying the script
    p = argparse.ArgumentParser(
        description="Iteration 2: Feature selection and re-evaluation."  # sets the description shown in the help message when the user runs the script with --help
    )
    p.add_argument(
        "--threshold", type=float, default=0.01,
        help="Mean importance threshold for feature selection (default: 0.01). "
             "Features with mean importance < threshold are dropped."  # defines the --threshold argument accepting a float value defaulting to 0.01, with a help message explaining its purpose
    )
    p.add_argument(
        "--testsize", type=float, default=0.2,
        help="Test set fraction for within-dataset split (default: 0.2)"  # defines the --testsize argument accepting a float value defaulting to 0.2, with a help message explaining its purpose
    )
    return p.parse_args()  # parses the command-line arguments and returns them as a Namespace object with threshold and testsize attributes


if __name__ == "__main__":
    # checks whether the script is being run directly rather than imported as a module, and if so, parses arguments and calls the main function
    try:
        args = parse_args()                                    # parses the command-line arguments to retrieve the threshold and testsize values
        main(threshold=args.threshold, test_size=args.testsize)  # calls the main function with the parsed threshold and test size arguments to execute the full Iteration 2 feature selection evaluation
    except Exception:
        log.error("Unhandled error:\n%s", traceback.format_exc())  # catches any unhandled exception, logs the full traceback at ERROR level so the user can diagnose what went wrong
        sys.exit(1)  # exits with a non-zero status code to indicate that the script terminated due to an unhandled error
