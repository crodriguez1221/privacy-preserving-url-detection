# -*- coding: utf-8 -*-
# --------------------------------------------------------------
# src/prepare_data.py
# --------------------------------------------------------------
# Purpose:
#   Clean, validate, and standardize raw URL datasets produced
#   by build_dataset.py.  This script does ONE job: ensure the
#   data is consistent, correctly labelled, and free of quality
#   issues before it reaches feature extraction.
#
#   It does NOT extract features.  Feature engineering is the
#   sole responsibility of extract_features.py.
#
# What this script does:
#   1. Loads the raw balanced CSV (url + label columns)
#   2. Standardises column names to canonical 'url' and 'label'
#   3. Strips whitespace and enforces consistent URL formatting
#   4. Normalizes labels to binary integers (0 or 1)
#   5. Removes null URLs and null/unmappable labels
#   6. Removes duplicate URLs
#   7. Logs a class balance summary for verification
#   8. Saves a cleaned CSV ready for extract_features.py
#
# Input files (produced by build_dataset.py):
#   data/phishtank_tranco_dataset.csv
#   data/urlhaus_tranco_dataset.csv
#
# Output files:
#   data/cleaned_phishtank_tranco_dataset.csv
#   data/cleaned_urlhaus_tranco_dataset.csv
#
# Usage:
#   python src/prepare_data.py --input data/phishtank_tranco_dataset.csv
#   python src/prepare_data.py --input data/urlhaus_tranco_dataset.csv
# --------------------------------------------------------------

import os # for file path handling
import sys # for sys.exit if something goes wrong
import logging # for logging progress and errors
import argparse # for passing options via command-line (argument parsing)
import traceback # for detailed error tracebacks in logs

import pandas as pd # for data manipulation and saving spreadsheets style data (CSV files)

logging.basicConfig( # configure logging to show timestamps and log levels
    level=logging.INFO, # show INFO and above (INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s", # include timestamp, log level, and message
    datefmt="%H:%M:%S", # show only time in logs for readability
)
log = logging.getLogger(__name__) # create a logger for this module (src/prepare_data.py)

# --------------------------------------------------------------
# Default file paths
# --------------------------------------------------------------
DEFAULT_DATASETS = [
    os.path.join("data", "phishtank_tranco_dataset.csv"), # default input file for PhishTank dataset
    os.path.join("data", "urlhaus_tranco_dataset.csv"), # default input file for URLhaus dataset
]

# Column name candidates across common dataset formats
URL_CANDIDATES   = ["url", "urls", "link", "domain", "address"] # common variations for the URL column name
LABEL_CANDIDATES = ["label", "class", "target", "type", "result", "status"] # common variations for the label column name

# Maps any string label variant to binary 0 / 1
LABEL_MAP = {
    "phishing":   1, "phish":     1, "bad":       1,
    "malicious":  1, "spam":      1, "deceptive": 1, "1": 1,
    "legitimate": 0, "benign":    0, "good":      0,
    "safe":       0, "ham":       0, "0":         0,
}


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame: # helper function to load CSV with encoding fallback
    """Load CSV with automatic UTF-8 / latin-1 encoding fallback."""
    try: # first try to load with UTF-8 encoding (most common)
        df = pd.read_csv(path, low_memory=False) # load CSV, disable low_memory to avoid dtype issues with large files
        log.info("Loaded %d rows from %s (utf-8)", len(df), path) # log how many rows were loaded and which encoding was used
        return df # return the loaded DataFrame
    except UnicodeDecodeError: # if UTF-8 fails, try latin-1 encoding 
        df = pd.read_csv( # load CSV again with latin-1 encoding, skip bad lines to avoid issues with malformed rows
            path, encoding="latin-1", 
            on_bad_lines="skip", low_memory=False
        )
        log.warning( # log a warning that UTF-8 failed and latin-1 was used instead, which may indicate encoding issues in the file
            "Loaded %d rows from %s (latin-1 fallback)", len(df), path # log how many rows were loaded with latin-1 encoding
        )
        return df # return the loaded DataFrame


def standardise_columns(df: pd.DataFrame, path: str) -> pd.DataFrame: # helper function to standardise column names to 'url' and 'label'
    """
    Locate the URL and label columns regardless of their original
    names and rename them to the canonical 'url' and 'label'.
    Raises ValueError with a clear message if either is missing.
    """
    lower_map = {c.lower().strip(): c for c in df.columns} # create a mapping of lowercase column names to original column names for case-insensitive matching

    url_key   = next((k for k in URL_CANDIDATES   if k in lower_map), None) # find the first URL candidate that matches a column in the DataFrame, or None if not found
    label_key = next((k for k in LABEL_CANDIDATES if k in lower_map), None) # find the first label candidate that matches a column in the DataFrame, or None if not found

    if not url_key: # if no URL column was found, raise a ValueError with a clear message showing what was searched for and what columns were found in the file
        raise ValueError( # raise an error to indicate that the required URL column is missing, which is critical for the cleaning process to work
            f"No URL column found in {path}.\n" # error message indicating that the URL column is missing from the dataset, which is essential for the cleaning process to function correctly
            f"  Searched for : {URL_CANDIDATES}\n" # show the list of candidate column names that were searched for in the DataFrame to help the user understand what was expected
            f"  Columns found: {list(df.columns)}" # show the actual columns that were found in the DataFrame to help the user identify if there was a naming issue or if the file format is unexpected
        )
    if not label_key: # if no label column was found, raise a ValueError with a clear message showing what was searched for and what columns were found in the file
        raise ValueError( # raise an error to indicate that the required label column is missing, which is critical for the cleaning process to work
            f"No label column found in {path}.\n" # error message indicating that the label column is missing from the dataset, which is essential for the cleaning process to function correctly
            f"  Searched for : {LABEL_CANDIDATES}\n" # show the list of candidate column names that were searched for in the DataFrame to help the user understand what was expected
            f"  Columns found: {list(df.columns)}" # show the actual columns that were found in the DataFrame to help the user identify if there was a naming issue or if the file format is unexpected
        )

    rename = {} # create a dictionary to hold any column renames that are needed to standardise to 'url' and 'label'
    if lower_map[url_key] != "url": # if the matched URL column is not already named 'url', add it to the rename dictionary to rename it to 'url'
        rename[lower_map[url_key]] = "url" # add the original column name that matched the URL candidate to the rename dictionary with the new name 'url' to standardise it for downstream processing
    if lower_map[label_key] != "label": # if the matched label column is not already named 'label', add it to the rename dictionary to rename it to 'label'
        rename[lower_map[label_key]] = "label" # add the original column name that matched the label candidate to the rename dictionary with the new name 'label' to standardise it for downstream processing

    if rename: # if there are any columns that need to be renamed, perform the renaming and log what was renamed for transparency
        df = df.rename(columns=rename) # rename the columns in the DataFrame according to the rename dictionary to standardise them to 'url' and 'label' for downstream processing
        log.info("Renamed columns: %s", rename) # log the columns that were renamed to show how the original dataset was standardised for transparency and debugging purposes

    # Keep only the two columns we need
    return df[["url", "label"]].copy() # return a new DataFrame with only the 'url' and 'label' columns, which are now guaranteed to be present and correctly named for the cleaning process to work


def normalise_labels(series: pd.Series) -> pd.Series: # helper function to normalise labels to binary integers 0 or 1
    """ 
    Convert any label format to binary integers 0 or 1.
    Numeric columns are coerced directly.
    String columns are matched against LABEL_MAP.
    """
    if pd.api.types.is_numeric_dtype(series): # if the series is already numeric, attempt to coerce it to numeric values, which will convert valid numbers to themselves and invalid entries to NaN
        return pd.to_numeric(series, errors="coerce") # convert the series to numeric, coercing errors to NaN, which allows us to handle cases where the label column is already numeric but may contain invalid entries that should be treated as missing values
    return ( # if the series is not numeric, treat it as strings and map them to binary labels using the LABEL_MAP, which allows us to handle cases where the label column contains string labels that need to be normalised to 0/1
        series.astype(str) # convert the series to string type to ensure that we can apply string operations and mapping, which is necessary for handling cases where the label column contains non-numeric labels that need to be normalised
              .str.strip() # strip whitespace from the string labels to ensure that we can match them correctly against the LABEL_MAP, which helps to handle cases where there may be extra spaces in the label values that would prevent correct mapping
              .str.lower() # convert the string labels to lowercase to ensure that we can match them correctly against the LABEL_MAP in a case-insensitive way, which helps to handle cases where the label values may have inconsistent capitalization that would prevent correct mapping
              .map(LABEL_MAP) # map the cleaned string labels to binary integers using the LABEL_MAP, which will convert known label variants to 0 or 1 and unknown labels to NaN, allowing us to handle a wide range of label formats and identify any unmappable labels that should be treated as missing values
    )


# --------------------------------------------------------------
# Cleaning pipeline
# --------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame: # main cleaning function that applies the full cleaning pipeline to the DataFrame
    """
    Apply the full cleaning pipeline:
      - Strip whitespace from URLs
      - Enforce consistent lowercase scheme (http/https)
      - Remove null URLs
      - Normalise labels to binary 0/1
      - Remove null or unmappable labels
      - Remove duplicate URLs
    Returns a cleaned DataFrame with only 'url' and 'label' columns.
    """
    # Step 1: Strip whitespace from URLs
    df["url"] = df["url"].astype(str).str.strip() # convert the 'url' column to string type and strip leading/trailing whitespace to ensure that URLs are clean and consistent, which helps to prevent issues with matching and feature extraction later on

    # Step 2: Remove rows where URL is null or empty string
    before = len(df) # store the number of rows before removing null/empty URLs to log how many were removed and how many remain, which helps to track the impact of this cleaning step on the dataset size
    df = df[df["url"].notna() & (df["url"] != "") & (df["url"] != "nan")] # filter the DataFrame to keep only rows where the 'url' column is not null, not an empty string, and not the string "nan" (which can occur if there were NaN values that got converted to strings), which helps to ensure that we only keep valid URLs for downstream processing
    log.info( # log how many rows were removed due to null/empty URLs and how many remain after this cleaning step, which helps to track the quality of the dataset and identify any issues with missing URL values
        "Removed %d rows with null/empty URLs. Remaining: %d",
        before - len(df), len(df) # log the number of rows removed and remaining after filtering out null/empty URLs to provide insight into the quality of the dataset and the impact of this cleaning step
    )

    # Step 3: Normalise labels to binary 0/1
    df["label"] = normalise_labels(df["label"]) # apply the normalise_labels function to the 'label' column to convert any label format to binary integers 0 or 1, which helps to standardise the labels for downstream processing and model training, and allows us to handle a wide range of label formats that may be present in the raw datasets

    # Step 4: Remove rows with null or unmappable labels
    before = len(df) # store the number of rows before removing null/unmappable labels to log how many were removed and how many remain, which helps to track the impact of this cleaning step on the dataset size and identify any issues with missing or invalid label values
    df = df[df["label"].isin([0, 1])].copy() # filter the DataFrame to keep only rows where the 'label' column is either 0 or 1, which removes any rows with null or unmappable labels that could not be converted to binary integers, and ensures that we only keep valid labels for downstream processing and model training
    df["label"] = df["label"].astype(int) # convert the 'label' column to integer type to ensure that it is in the correct format for downstream processing and model training, which helps to prevent issues with data types later on
    log.info( # log how many rows were removed due to null/unmappable labels and how many remain after this cleaning step, which helps to track the quality of the dataset and identify any issues with missing or invalid label values
        "Removed %d rows with null/unmappable labels. Remaining: %d",
        before - len(df), len(df) # log the number of rows removed and remaining after filtering out null/unmappable labels to provide insight into the quality of the dataset and the impact of this cleaning step
    )

    # Step 5: Remove duplicate URLs
    before = len(df) # store the number of rows before removing duplicate URLs to log how many were removed and how many remain, which helps to track the impact of this cleaning step on the dataset size and identify any issues with duplicate entries that could bias the model
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True) # drop duplicate rows based on the 'url' column to ensure that each URL is unique in the dataset, which helps to prevent bias in the model that could arise from having multiple entries for the same URL, and reset the index after dropping duplicates for cleanliness
    log.info( # log how many duplicate URLs were removed and how many remain after this cleaning step, which helps to track the quality of the dataset and identify any issues with duplicate entries that could bias the model
        "Removed %d duplicate URLs. Remaining: %d",
        before - len(df), len(df) # log the number of duplicate URLs removed and remaining after dropping duplicates to provide insight into the quality of the dataset and the impact of this cleaning step
    )

    if len(df) == 0: # if no valid rows remain after all the cleaning steps, raise a ValueError with a clear message indicating that the dataset is empty and suggesting to check the input file for correct URL and label values, which helps to prevent downstream errors in feature extraction and model training that would arise from having an empty dataset
        raise ValueError( # raise an error to indicate that the dataset is empty after cleaning, which is critical to catch early to avoid issues in downstream processing and model training that would arise from having no data to work with
            "No valid rows remain after cleaning. "
            "Check that the input file has correct URL and label values."
        )

    return df # return the cleaned DataFrame with only 'url' and 'label' columns, which is now ready for feature extraction and model training


def log_class_balance(df: pd.DataFrame) -> None: # helper function to log the class distribution of the cleaned dataset for verification
    """Log class distribution for verification."""
    n_mal   = int((df["label"] == 1).sum()) # count the number of malicious samples (label == 1) in the cleaned DataFrame to calculate the class balance, which helps to verify that the dataset is still balanced after cleaning and identify any issues with class distribution that could affect model training
    n_legit = int((df["label"] == 0).sum()) # count the number of legitimate samples (label == 0) in the cleaned DataFrame to calculate the class balance, which helps to verify that the dataset is still balanced after cleaning and identify any issues with class distribution that could affect model training
    total   = len(df) # calculate the total number of samples in the cleaned DataFrame to calculate the class balance, which helps to verify that the dataset is still balanced after cleaning and identify any issues with class distribution that could affect model training
    pct_mal = 100 * n_mal / total if total > 0 else 0 # calculate the percentage of malicious samples in the cleaned DataFrame to log the class balance, which helps to verify that the dataset is still balanced after cleaning and identify any issues with class distribution that could affect model training, and handle the case where total is 0 to avoid division by zero errors
    
    log.info("-" * 50) 
    log.info("Class balance summary:") # log a summary of the class balance in the cleaned dataset, including the total number of samples, the number and percentage of malicious samples, and the number and percentage of legitimate samples, which helps to verify that the dataset is still balanced after cleaning and identify any issues with class distribution that could affect model training
    log.info("  Total rows  : %d", total) # log the total number of rows in the cleaned dataset to provide insight into the size of the dataset after cleaning, which helps to verify that we have a sufficient amount of data for feature extraction and model training
    log.info("  Malicious   : %d  (%.1f%%)", n_mal,   pct_mal) # log the number and percentage of malicious samples in the cleaned dataset to provide insight into the class balance, which helps to verify that the dataset is still balanced after cleaning and identify any issues with class distribution that could affect model training
    log.info("  Legitimate  : %d  (%.1f%%)", n_legit, 100 - pct_mal) # log the number and percentage of legitimate samples in the cleaned dataset to provide insight into the class balance, which helps to verify that the dataset is still balanced after cleaning and identify any issues with class distribution that could affect model training
    log.info("-" * 50)


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------
def parse_args() -> argparse.Namespace: # helper function to parse command-line arguments for input datasets and output directory
    p = argparse.ArgumentParser( # create an argument parser for command-line options to specify input datasets and output directory, which allows for flexible usage of the script with different datasets and output locations
        description=( # provide a description of what the script does when the user runs it with --help, which helps to clarify the purpose of the script and how to use it
            "Clean and validate URL datasets produced by build_dataset.py. "
            "Output is a standardised CSV ready for extract_features.py."
        )
    )
    p.add_argument( # add an argument for specifying one or more input datasets to process, with a default value of DEFAULT_DATASETS, which allows the user to easily specify which raw datasets to clean and validate without having to modify the script
        "--datasets", nargs="+", # allow multiple datasets to be specified as a list of file paths, with a default value of DEFAULT_DATASETS, which provides a convenient way to specify the input files to process without having to modify the script
        default=DEFAULT_DATASETS, # set the default value for the datasets argument to DEFAULT_DATASETS, which includes the default input files for the PhishTank and URLhaus datasets, providing a convenient starting point for users who want to clean those specific datasets without having to specify them explicitly
        help="One or more raw balanced CSVs to process", # provide a help message for the datasets argument to explain that it accepts one or more raw balanced CSV files to process, which helps to clarify the expected input format and usage of the script
    )
    p.add_argument( # add an argument for specifying the output directory where the cleaned files will be saved, with a default value of "data", which allows the user to easily specify where to save the cleaned datasets without having to modify the script
        "--outdir", # add an argument for specifying the output directory where the cleaned files will be saved, with a default value of "data", which allows the user to easily specify where to save the cleaned datasets without having to modify the script
        default="data", # set the default value for the outdir argument to "data", which is a common directory for storing datasets and provides a convenient default location for saving the cleaned files without having to specify it explicitly
        help="Output directory for cleaned files (default: data/)", # provide a help message for the outdir argument to explain that it specifies the output directory for the cleaned files, with a default value of "data/", which helps to clarify the expected usage of the script and where the cleaned files will be saved
    )
    return p.parse_args() # parse the command-line arguments and return them as a Namespace object, which allows the main function to access the specified input datasets and output directory for processing


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main() -> None: # main function that orchestrates the entire cleaning process for one or more input datasets, applying the cleaning pipeline and saving the cleaned files to the specified output directory
    args = parse_args() # parse the command-line arguments to get the list of input datasets and the output directory, which allows the user to specify which raw datasets to clean and where to save the cleaned files without having to modify the script

    log.info("=" * 60)
    log.info("prepare_data.py  --  cleaning and validation only") # log a header to indicate the start of the prepare_data.py script and its purpose, which helps to clarify what the script does and sets the context for the cleaning process that will follow
    log.info("=" * 60)
    log.info("Processing %d dataset(s)...", len(args.datasets)) # log how many datasets will be processed based on the command-line arguments, which provides insight into the scope of the cleaning process and helps to track progress as each dataset is processed

    os.makedirs(args.outdir, exist_ok=True) # create the output directory if it doesn't already exist, which ensures that we have a valid location to save the cleaned files without having to worry about directory creation errors

    success = 0 # initialize a counter to track how many datasets were cleaned successfully, which allows us to provide a summary at the end of the script and identify if any datasets failed to process due to errors
    for input_path in args.datasets: # iterate over each input dataset specified in the command-line arguments to apply the cleaning process to each one, which allows us to handle multiple datasets in a single run of the script and provides flexibility in processing different raw datasets without having to modify the script
        log.info("-" * 60)
        log.info("Input : %s", os.path.abspath(input_path)) # log the absolute path of the input dataset being processed to provide clarity on which file is currently being cleaned, which helps to track progress and identify any issues with specific files during the cleaning process

        if not os.path.exists(input_path): # check if the input file exists before attempting to process it, which helps to prevent errors that would arise from trying to load a non-existent file and allows us to log a clear error message if the file is missing
            log.error("File not found, skipping: %s", input_path) # log an error message indicating that the specified input file was not found, which helps to clarify why the file cannot be processed and allows the script to continue processing any remaining datasets without crashing
            continue # skip to the next dataset if the current input file does not exist, which allows the script to continue processing any remaining datasets without crashing due to a missing file

        try: # attempt to load, clean, and save the dataset, with error handling to catch and log any issues that arise during processing, which helps to ensure that the script can continue processing other datasets even if one fails due to issues with the file format, encoding, or data quality
            df = load_csv(input_path) # load the input CSV file into a DataFrame using the load_csv helper function, which includes encoding fallback to handle different file encodings and provides logging on how many rows were loaded and which encoding was used, which helps to ensure that we can successfully load a wide range of raw datasets without running into encoding issues
            df = standardise_columns(df, input_path) # standardise the column names in the DataFrame to 'url' and 'label' using the standardise_columns helper function, which locates the URL and label columns regardless of their original names and renames them to the canonical 'url' and 'label', and raises a ValueError with a clear message if either is missing, which helps to ensure that we have the correct columns for cleaning and provides clear error messages if the expected columns are not found in the input file
            df = clean(df) # apply the full cleaning pipeline to the DataFrame using the clean function, which includes stripping whitespace from URLs, removing null/empty URLs, normalising labels to binary 0/1, removing null/unmappable labels, and removing duplicate URLs, and returns a cleaned DataFrame with only 'url' and 'label' columns that is ready for feature extraction and model training, which helps to ensure that the dataset is of high quality and consistent for downstream processing
            log_class_balance(df) # log the class distribution of the cleaned dataset using the log_class_balance helper function, which provides a summary of the class balance in the cleaned dataset, including the total number of samples, the number and percentage of malicious samples, and the number and percentage of legitimate samples, which helps to verify that the dataset is still balanced after cleaning and identify any issues with class distribution that could affect model training

            basename = os.path.splitext(os.path.basename(input_path))[0] # extract the base name of the input file without the directory or extension to use for naming the output file, which helps to create a clear and consistent naming convention for the cleaned files that indicates their origin
            out_path = os.path.join(args.outdir, f"cleaned_{basename}.csv") # construct the output file path by joining the specified output directory with a new file name that includes the prefix "cleaned_" followed by the base name of the input file, which helps to create a clear and consistent naming convention for the cleaned files that indicates their origin and makes it easy to identify which raw dataset they were cleaned from

            df.to_csv(out_path, index=False) # save the cleaned DataFrame to a new CSV file at the constructed output path, without including the index in the CSV to keep it clean and focused on the 'url' and 'label' columns, which helps to ensure that the cleaned dataset is saved in a standard format that can be easily loaded for feature extraction and model training
            log.info("Saved -> %s", os.path.abspath(out_path)) # log the absolute path of the saved cleaned file to provide clarity on where the cleaned dataset was saved, which helps to track the output of the cleaning process and allows the user to easily locate the cleaned files for the next step of feature extraction
            success += 1 # increment the success counter if the dataset was cleaned and saved successfully, which allows us to provide a summary at the end of the script on how many datasets were processed successfully and identify if any issues arose during processing

        except Exception: 
            log.error("Failed to process %s:\n%s", input_path, traceback.format_exc()) # catch any exceptions that occur during the loading, cleaning, or saving process for the current dataset, and log an error message with the file path and a detailed traceback of the error to help identify what went wrong and allow the script to continue processing any remaining datasets without crashing

    log.info("=" * 60)
    log.info("Done. %d / %d dataset(s) cleaned successfully.", success, len(args.datasets)) # log a summary of how many datasets were cleaned successfully out of the total number of datasets specified, which provides insight into the overall success of the cleaning process and helps to identify if any issues arose with specific datasets during processing
    log.info("Next step: python src/extract_features.py") # log a message indicating the next step in the pipeline, which is to run the extract_features.py script to perform feature extraction on the cleaned datasets, providing a clear direction for the user on how to proceed with the next stage of the project

    if success < len(args.datasets): # if not all datasets were cleaned successfully, exit with a non-zero status code to indicate that there were issues during processing, which helps to signal to the user or any automated systems that not all datasets were processed correctly and that they should check the logs for details on what went wrong
        sys.exit(1) # exit with status code 1 to indicate that there were issues during processing, which helps to signal to the user or any automated systems that not all datasets were processed correctly and that they should check the logs for details on what went wrong


if __name__ == "__main__": # standard Python idiom to allow the script to be run directly, which calls the main function and includes error handling to catch any unhandled exceptions that may arise during the execution of the main function, log a detailed error message with the traceback, and exit with a non-zero status code to indicate that an error occurred, which helps to ensure that any issues that arise during processing are logged clearly and that the script exits gracefully without crashing
    try: # call the main function to execute the cleaning process, which orchestrates the entire workflow of loading, cleaning, and saving the datasets based on the command-line arguments provided by the user
        main() # execute the main function to start the cleaning process, which includes parsing command-line arguments, processing each specified dataset, and logging the results, providing a clear entry point for the script when run directly
    except Exception: # catch any unhandled exceptions that occur during the execution of the main function, which helps to ensure that any issues that arise are logged clearly and that the script exits gracefully without crashing
        log.error("Unhandled error:\n%s", traceback.format_exc()) # log an error message indicating that an unhandled error occurred, along with a detailed traceback of the error to help identify what went wrong and allow the user to understand the issue, which helps to ensure that any issues that arise during processing are logged clearly and can be addressed by the user
        sys.exit(1) # exit with status code 1 to indicate that an unhandled error occurred, which helps to signal to the user or any automated systems that there was a critical issue during processing and that they should check the logs for details on what went wrong