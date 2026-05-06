# -*- coding: utf-8 -*-
# --------------------------------------------------------------
# src/extract_features.py
# --------------------------------------------------------------
# Purpose:
#   Engineer a standardised 24-feature set from cleaned URL
#   datasets produced by prepare_data.py.  This script does
#   ONE job: feature extraction.
#
#   It reads cleaned CSVs (url + label only) and outputs
#   feature-rich CSVs ready for train_models.py.
#
#   All 24 features are derived purely from the URL string
#   itself — no external lookups, no DNS queries, no WHOIS.
#   This guarantees that BOTH datasets produce IDENTICAL
#   feature column names, which is a requirement for
#   cross-dataset generalisation testing.
#
# Feature groups (24 total):
#   A) Length measurements  (4)  : url_length, hostname_length,
#                                   path_length, query_length
#   B) Character counts    (10)  : num_dots, num_hyphens,
#                                   num_underscores, num_slashes,
#                                   num_qmarks, num_equals,
#                                   num_ampersands, num_at,
#                                   num_percent, num_digits
#   C) Structural/boolean   (6)  : has_https, has_ip, has_port,
#                                   has_www, has_at_in_url,
#                                   subdomain_depth
#   D) Entropy & ratios     (4)  : url_entropy, hostname_entropy,
#                                   digit_ratio, special_char_ratio
#
# Input files (produced by prepare_data.py):
#   data/cleaned_phishtank_tranco_dataset.csv
#   data/cleaned_urlhaus_tranco_dataset.csv
#
# Output files:
#   data/features_phishtank_tranco_dataset.csv
#   data/features_urlhaus_tranco_dataset.csv
#
# Usage:
#   python src/extract_features.py
#   python src/extract_features.py --input data/cleaned_phishtank_tranco_dataset.csv
#   python src/extract_features.py --datasets data/cleaned_phishtank_tranco_dataset.csv data/cleaned_urlhaus_tranco_dataset.csv
# --------------------------------------------------------------

import os # for file path handling
import sys # for sys.exit if something goes wrong
import logging # for logging progress and errors
import argparse # for passing options via command-line (argument parsing)
import traceback # for detailed error tracebacks in logs
from urllib.parse import urlparse # for parsing URLs into components (hostname, path, query, etc.)

import pandas as pd # for data manipulation and saving spreadsheets style data (CSV files)
import numpy as np # for numerical operations and array handling

logging.basicConfig( # configure logging to show timestamps and log levels
    level=logging.INFO, # show INFO and above (INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s", # include timestamp, log level, and message
    datefmt="%H:%M:%S", # show only time in logs for readability
)
log = logging.getLogger(__name__) # create a logger for this module (src/extract_features.py)

# --------------------------------------------------------------
# Default dataset paths
# (output of prepare_data.py)
# --------------------------------------------------------------
DEFAULT_DATASETS = [ 
    os.path.join("data", "cleaned_phishtank_tranco_dataset.csv"), # default path to the cleaned PhishTank dataset CSV
    os.path.join("data", "cleaned_urlhaus_tranco_dataset.csv"), # default path to the cleaned URLhaus dataset CSV
]


# --------------------------------------------------------------
# Loader
# --------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame: # load a CSV file into a DataFrame, with UTF-8 encoding and fallback to latin-1 if needed
    """Load a cleaned CSV with UTF-8 / latin-1 encoding fallback."""
    try: # first try to load with UTF-8 encoding (the most common encoding for modern CSVs)
        df = pd.read_csv(path, low_memory=False) # load the CSV file at 'path' into a DataFrame, with low_memory=False to avoid dtype warnings for large files
        log.info("Loaded %d rows from %s", len(df), path) # log how many rows were loaded successfully
        return df # return the loaded DataFrame to the caller
    except UnicodeDecodeError: # if there was a UnicodeDecodeError (common when the file is not UTF-8 encoded), try again with latin-1 encoding
        df = pd.read_csv( # load the CSV file again, but this time with encoding="latin-1" to handle files that are not UTF-8 encoded
            path, encoding="latin-1", # skip bad lines that can't be parsed, and low_memory=False to avoid dtype warnings
            on_bad_lines="skip", low_memory=False # log a warning that we had to fall back to latin-1 encoding, which may indicate some encoding issues with the file
        )
        log.warning( # log a warning that we had to fall back to latin-1 encoding, which may indicate some encoding issues with the file
            "Loaded %d rows from %s (latin-1 fallback)", len(df), path
        )
        return df # return the loaded DataFrame to the caller


# --------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------
def _shannon_entropy(s: str) -> float: # compute the Shannon entropy of a string, which is a measure of randomness or unpredictability in the characters of the string
    """
    Compute character-level Shannon entropy for a string.
    Higher entropy indicates more randomness — a common trait
    of algorithmically generated phishing URLs.
    """
    if not s: # if the string is empty or None, return 0.0 entropy (no randomness)
        return 0.0 # compute the frequency of each character in the string, normalized to sum to 1 (probability distribution of characters)
    freq = pd.Series(list(s)).value_counts(normalize=True) # create a Series of character frequencies by counting each character in the string and normalizing to get probabilities
    return float(-(freq * np.log2(freq)).sum()) # compute the Shannon entropy using the formula: H = -sum(p(x) * log2(p(x))) for each character, and return it as a float


def extract_features(df: pd.DataFrame) -> pd.DataFrame: # extract 24 features from the 'url' column of the input DataFrame, and return a new DataFrame with the original 'url' and 'label' columns plus the new feature columns
    """
    Engineer all 24 URL-string features from the 'url' column.

    Input:  DataFrame with columns ['url', 'label']
    Output: DataFrame with columns ['url', 'label', <24 features>]

    No external data sources are used — all features are derived
    directly from the URL string, ensuring both datasets produce
    identical column schemas for cross-dataset evaluation.
    """
    df = df.copy() # work on a copy of the input DataFrame to avoid modifying the original
    df["url"] = df["url"].astype(str).str.strip() # ensure the 'url' column is treated as strings and remove leading/trailing whitespace

    log.info("Extracting features from %d URLs ...", len(df)) # log how many URLs we are extracting features from

    # Parse URL components once for reuse across feature groups
    def safe_parse(u): # helper function to parse a URL string safely, adding a default scheme if missing to avoid parsing errors
        return urlparse(u if "://" in u else "http://" + u) # if the URL does not contain "://", prepend "http://" to ensure it can be parsed correctly by urlparse, which expects a scheme (like http or https) to identify the hostname and other components properly

    parsed    = df["url"].apply(safe_parse) # apply the safe_parse function to each URL in the 'url' column, resulting in a Series of ParseResult objects that contain the parsed components of each URL (scheme, hostname, path, query, etc.)
    hostname  = parsed.apply(lambda p: p.hostname or "") # extract the hostname from the parsed URL, using an empty string if the hostname is None (which can happen if the URL is malformed or missing a hostname)
    path_part = parsed.apply(lambda p: p.path or "") # extract the path from the parsed URL, using an empty string if the path is None (which can happen if the URL is malformed or missing a path)
    query     = parsed.apply(lambda p: p.query or "") # extract the query string from the parsed URL, using an empty string if the query is None (which can happen if the URL does not have a query component)

    # ── GROUP A: Length measurements ────────────────────────────
    df["url_length"]      = df["url"].str.len() # compute the length of the entire URL string and store it in the 'url_length' column
    df["hostname_length"] = hostname.str.len() # compute the length of the hostname component and store it in the 'hostname_length' column
    df["path_length"]     = path_part.str.len() # compute the length of the path component and store it in the 'path_length' column
    df["query_length"]    = query.str.len() # compute the length of the query string and store it in the 'query_length' column

    # ── GROUP B: Character count features ───────────────────────
    df["num_dots"]        = df["url"].str.count(r"\.") # count the number of dots (.) in the URL and store it in the 'num_dots' column
    df["num_hyphens"]     = df["url"].str.count(r"\-") # count the number of hyphens (-) in the URL and store it in the 'num_hyphens' column
    df["num_underscores"] = df["url"].str.count(r"\_") # count the number of underscores (_) in the URL and store it in the 'num_underscores' column
    df["num_slashes"]     = df["url"].str.count(r"\/") # count the number of slashes (/) in the URL and store it in the 'num_slashes' column
    df["num_qmarks"]      = df["url"].str.count(r"\?") # count the number of question marks (?) in the URL and store it in the 'num_qmarks' column
    df["num_equals"]      = df["url"].str.count(r"=") # count the number of equals signs (=) in the URL and store it in the 'num_equals' column
    df["num_ampersands"]  = df["url"].str.count(r"&") # count the number of ampersands (&) in the URL and store it in the 'num_ampersands' column
    df["num_at"]          = df["url"].str.count(r"@") # count the number of at signs (@) in the URL and store it in the 'num_at' column
    df["num_percent"]     = df["url"].str.count(r"%") # count the number of percent signs (%) in the URL and store it in the 'num_percent' column
    df["num_digits"]      = df["url"].str.count(r"\d") # count the number of digits (0-9) in the URL and store it in the 'num_digits' column

    # ── GROUP C: Structural and boolean features ─────────────────
    df["has_https"] = parsed.apply( # check if the URL scheme is "https" and store it as a binary feature (1 for https, 0 for anything else)
        lambda p: int(p.scheme == "https") # convert the boolean result to an integer (1 for True, 0 for False) and store it in the 'has_https' column
    )
    df["has_ip"] = hostname.str.match( # check if the hostname matches the pattern of an IPv4 address (e.g.,
        r"^\d{1,3}(\.\d{1,3}){3}$" # regex pattern for matching an IPv4 address, which consists of four groups of 1 to 3 digits separated by dots) and store it as a binary feature (1 for IP address, 0 for hostname)
    ).astype(int) # convert the boolean result to an integer (1 for True, 0 for False) and store it in the 'has_ip' column
    df["has_port"] = parsed.apply( # check if the parsed URL has a port number specified (p.port is not None) and store it as a binary feature (1 for has port, 0 for no port)
        lambda p: int(bool(p.port)) # convert the presence of a port number to an integer (1 if p.port is not None, 0 if p.port is None) and store it in the 'has_port' column
    ).astype(int)
    df["has_www"] = hostname.str.startswith("www.").astype(int) #
    df["has_at_in_url"] = df["url"].str.contains( # check if the URL contains an at sign (@) anywhere in the URL string and store it as a binary feature (1 for contains @, 0 for does not contain @)
        "@", regex=False # check for literal "@" character without using regex, and convert the boolean result to an integer (1 for True, 0 for False) and store it in the 'has_at_in_url' column
    ).astype(int) # compute the subdomain depth by counting the number of dots in the hostname and subtracting 2 (for the main domain and TLD), ensuring it does not go below 0

    def subdomain_depth(h): # helper function to compute the subdomain depth from the hostname by counting the number of dots and subtracting 2 (for the main domain and TLD), ensuring it does not go below 0
        parts = h.split(".") if h else [] # split the hostname into parts by dots, resulting in a list of subdomain components; if the hostname is empty or None, use an empty list
        return max(0, len(parts) - 2) # compute the subdomain depth as the number of parts minus 2 (for the main domain and TLD), and ensure it is not negative by using max(0, ...)

    df["subdomain_depth"] = hostname.apply(subdomain_depth) # apply the subdomain_depth function to the hostname to compute the subdomain depth for each URL and store it in the 'subdomain_depth' column

    # ── GROUP D: Entropy and ratio features ──────────────────────
    df["url_entropy"]      = df["url"].apply(_shannon_entropy) # compute the Shannon entropy of the entire URL string using the _shannon_entropy function and store it in the 'url_entropy' column
    df["hostname_entropy"] = hostname.apply(_shannon_entropy) # compute the Shannon entropy of the hostname component using the _shannon_entropy function and store it in the 'hostname_entropy' column

    url_len_safe = df["url_length"].replace(0, np.nan) # replace any zero lengths in 'url_length' with NaN to avoid division by zero when calculating ratios; this will result in NaN for the ratio features if the URL length is zero, which we will handle later by filling NaN with 0

    df["digit_ratio"] = df["num_digits"] / url_len_safe # calculate the ratio of digits to total URL length by dividing 'num_digits' by 'url_length', using url_len_safe to avoid division by zero; this will give us a measure of how many characters in the URL are digits, which can be a sign of algorithmically generated URLs if the ratio is high

    df["special_char_ratio"] = ( # calculate the ratio of special characters to total URL length by summing the counts of special characters and dividing by 'url_length', using url_len_safe to avoid division by zero; this will give us a measure of how many characters in the URL are special characters (like dots, hyphens, underscores, etc.), which can also be a sign of phishing URLs if the ratio is high
        df[["num_dots", "num_hyphens", "num_underscores", "num_at",
            "num_percent", "num_qmarks", "num_equals", "num_ampersands"]]
        .sum(axis=1) / url_len_safe # sum the counts of the specified special characters for each URL and divide by the URL length to get the ratio of special characters to total length
    )
    
    df = df.fillna(0) # fill any NaN values in the DataFrame with 0, which can occur in the ratio features if the URL length was zero (since we replaced zero lengths with NaN to avoid division by zero); this means that if a URL had a length of zero, its digit_ratio and special_char_ratio will be set to 0, which is a reasonable default since a zero-length URL cannot have any digits or special characters

    # ── Summary ───────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in ("url", "label")] # get a list of all the feature columns by excluding 'url' and 'label' from the DataFrame columns; this will be used for logging and for selecting the final output columns
    n_mal   = int((df["label"] == 1).sum()) # count the number of malicious URLs in the dataset by summing the number of rows where the 'label' column is 1 (indicating malicious), and convert it to an integer for logging
    n_legit = int((df["label"] == 0).sum()) # count the number of legitimate URLs in the dataset by summing the number of rows where the 'label' column is 0 (indicating legitimate), and convert it to an integer for logging
    pct     = 100 * n_mal / len(df) if len(df) > 0 else 0 # calculate the percentage of malicious URLs in the dataset by dividing n_mal by the total number of rows in the DataFrame and multiplying by 100; if the DataFrame is empty (length is 0), set the percentage to 0 to avoid division by zero

    log.info( # log the summary of the feature extraction process, including the number of rows and features extracted, and the class balance of malicious vs legitimate URLs in the dataset
        "Feature extraction complete: %d rows x %d features",
        len(df), len(feature_cols) # log the number of rows in the DataFrame and the number of feature columns extracted (excluding 'url' and 'label')
    )
    log.info( # log the class balance of malicious vs legitimate URLs in the dataset, showing the count and percentage of each class for transparency and to highlight any potential imbalance in the dataset
        "Class balance -> malicious: %d (%.1f%%)  legitimate: %d (%.1f%%)",
        n_mal, pct, n_legit, 100 - pct # log the count and percentage of malicious URLs (n_mal and pct) and legitimate URLs (n_legit and 100 - pct) in the dataset for transparency and to highlight any potential imbalance in the dataset
    )

    return df[["url", "label"] + feature_cols] # return a new DataFrame that includes the original 'url' and 'label' columns followed by all the extracted feature columns, ensuring a consistent column order for downstream processing in train_models.py


# --------------------------------------------------------------
# Process one dataset end-to-end
# --------------------------------------------------------------
def process_dataset(path: str, outdir: str) -> None: # process a single dataset by loading the cleaned CSV, extracting features, and saving the resulting feature-rich CSV to the specified output directory; this function encapsulates the entire end-to-end processing for one dataset
    """Load a cleaned CSV, extract features, and save."""
    if not os.path.exists(path): # check if the input file exists at the specified path; if it does not exist, log a warning and skip processing this dataset
        log.warning("File not found, skipping: %s", path) # log a warning that the file was not found at the specified path and that we are skipping processing for this dataset, which allows the script to continue processing any other datasets that do exist without crashing
        return # exit the function early since we cannot process a non-existent file

    log.info("=" * 60)
    log.info("Processing: %s", path) # log the path of the dataset we are currently processing, which helps track progress and identify which dataset is being worked on in the logs
    log.info("=" * 60)

    df = load_csv(path) # load the cleaned CSV file into a DataFrame using the load_csv function, which handles UTF-8 and latin-1 encoding and logs the number of rows loaded; this will be the input DataFrame for feature extraction

    if "url" not in df.columns or "label" not in df.columns: # check if the required columns 'url' and 'label' are present in the loaded DataFrame; if either column is missing, log an error and exit the function since we cannot proceed with feature extraction without these columns
        log.error( # log an error that the input file does not have the required 'url' and 'label' columns, and show the actual columns found in the DataFrame to help diagnose the issue; also suggest running prepare_data.py first to ensure the input file is in the correct format
            "Input file must have 'url' and 'label' columns. " # log an error that the input file must have 'url' and 'label' columns, which are required for feature extraction; also log the actual columns found in the DataFrame to help diagnose the issue, and suggest running prepare_data.py first to ensure the input file is in the correct format with 'url' and 'label' columns
            "Found: %s. Run prepare_data.py first.", list(df.columns) # log the actual columns found in the DataFrame to help diagnose the issue, and suggest running prepare_data.py first to ensure the input file is in the correct format with 'url' and 'label' columns
        )
        return # exit the function early since we cannot proceed with feature extraction without the required columns

    df = extract_features(df) # extract features from the loaded DataFrame using the extract_features function, which computes 24 features from the 'url' column and returns a new DataFrame with the original 'url' and 'label' columns plus the new feature columns; this will be the final DataFrame that we will save to a new CSV file

    # Output filename: features_<original_name>.csv
    # Strips the 'cleaned_' prefix since this is now the feature file
    basename = os.path.splitext(os.path.basename(path))[0] # get the base filename without the directory and extension from the input path, which will be used to construct the output filename; for example, if the input path is "data/cleaned_phishtank_tranco_dataset.csv", this will give us "cleaned_phishtank_tranco_dataset"
    basename = basename.replace("cleaned_", "") # remove the "cleaned_" prefix from the base filename to create a cleaner output filename that reflects the feature extraction step; for example, "cleaned_phishtank_tranco_dataset" will become "phishtank_tranco_dataset"
    out_path = os.path.join(outdir, f"features_{basename}.csv") # construct the output path by joining the specified output directory with the new filename that starts with "features_" followed by the modified base filename; for example, if outdir is "data" and basename is "phishtank_tranco_dataset", this will give us "data/features_phishtank_tranco_dataset.csv"

    os.makedirs(outdir, exist_ok=True) # create the output directory if it does not already exist, using exist_ok=True to avoid raising an error if the directory already exists; this ensures that we have a valid directory to save the output CSV file
    df.to_csv(out_path, index=False) # save the resulting DataFrame with features to a new CSV file at the constructed output path, without including the DataFrame index in the CSV file for cleaner output
    log.info( # log the success of saving the feature-rich CSV file, including the output path and the number of rows and columns in the resulting DataFrame, which helps confirm that the file was saved correctly and gives an overview of the size of the output dataset
        "Saved -> %s  (%d rows, %d columns)", # log the success of saving the feature-rich CSV file, including the output path and the number of rows and columns in the resulting DataFrame, which helps confirm that the file was saved correctly and gives an overview of the size of the output dataset
        out_path, len(df), len(df.columns) # log the output path where the feature-rich CSV file was saved, the number of rows in the resulting DataFrame, and the number of columns (features) in the resulting DataFrame for confirmation and overview of the output dataset
    )
    log.info("Next step: python src/train_models.py") # log a message suggesting the next step for the user, which is to run train_models.py to train machine learning models using the newly created feature-rich CSV files; this provides a clear call to action for the user after successfully extracting features


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------
def parse_args() -> argparse.Namespace: # parse command-line arguments for the script, allowing the user to specify one or more input dataset paths and an output directory for the feature CSVs; this function uses argparse to handle command-line arguments and provides default values and help messages for user guidance
    p = argparse.ArgumentParser( # create an ArgumentParser object to handle command-line arguments, with a description of the script's purpose and usage
        description=( # provide a description of the script's purpose and usage for the help message when the user runs the script with --help; this helps users understand what the script does and how to use it effectively
            "Extract 24 URL-string features from cleaned phishing datasets. "
            "Input must be produced by prepare_data.py."
        )
    )
    p.add_argument( # add an argument for specifying one or more input dataset paths, with a default value of DEFAULT_DATASETS and a help message explaining that these should be cleaned CSV paths to process; this allows users to specify which datasets they want to process without having to modify the script, and provides a default set of datasets for convenience
        "--datasets", nargs="+", # add an argument that can take one or more values (nargs="+") for the input dataset paths, allowing users to specify multiple datasets to process in one run; this is useful for processing both the PhishTank and URLhaus datasets together, or any other cleaned CSVs that follow the same format
        default=DEFAULT_DATASETS, # set the default value for the --datasets argument to DEFAULT_DATASETS, which is a list of the default cleaned CSV paths produced by prepare_data.py; this means that if the user does not specify any datasets on the command line, the script will automatically process these default datasets
        help="One or more cleaned CSV paths to process", # provide a help message for the --datasets argument to explain that it should be one or more cleaned CSV paths to process, which helps users understand what kind of input is expected for this argument
    )
    p.add_argument( # add an argument for specifying the output directory for the feature CSVs, with a default value of "data" and a help message explaining that this is where the feature CSVs will be saved; this allows users to control where the output files are saved without having to modify the script, and provides a sensible default for convenience
        "--outdir", # add an argument for specifying the output directory for the feature CSVs, allowing users to choose where they want the output files to be saved; this is useful for organizing outputs or saving to a different location if desired
        default="data", # set the default value for the --outdir argument to "data", which means that if the user does not specify an output directory on the command line, the script will save the feature CSVs to the "data" directory by default; this is a common convention for storing datasets and outputs in a project
        help="Output directory for feature CSVs (default: data/)", # provide a help message for the --outdir argument to explain that it is the output directory for the feature CSVs, and mention the default value of "data/" for clarity; this helps users understand what this argument does and where the output files will be saved if they do not specify a different directory
    )
    return p.parse_args() # parse the command-line arguments and return them as a Namespace object, which will contain the values for --datasets and --outdir that the user specified (or the default values if they did not specify them)


# --------------------------------------------------------------
# Entry point
# --------------------------------------------------------------
def main() -> None: # main function that serves as the entry point for the script, orchestrating the overall flow of loading datasets, extracting features, and saving the results; this function will be called when the script is run directly
    args = parse_args() # parse the command-line arguments using the parse_args function, which will give us access to the list of dataset paths and the output directory specified by the user (or the default values)

    log.info("=" * 60)
    log.info("extract_features.py  --  feature extraction only") 
    log.info("=" * 60)
    log.info(
        "Processing %d dataset(s)...", len(args.datasets) # log how many datasets we are going to process based on the length of the args.datasets list, which gives users an overview of the scope of the processing that will be done in this run
    )

    success = 0 # initialize a counter for the number of datasets processed successfully, which we will increment for each dataset that is processed without exceptions; this allows us to keep track of how many datasets were processed successfully and report it at the end
    for path in args.datasets: # loop through each dataset path specified in the args.datasets list, which allows us to process multiple datasets in one run; for each path, we will attempt to process it using the process_dataset function, and if any exceptions occur, we will catch them and log an error without crashing the entire script, allowing us to continue processing any remaining datasets
        try: # attempt to process each dataset in the list of dataset paths provided by the user (or the default list) using the process_dataset function, which will load the cleaned CSV, extract features, and save the resulting feature-rich CSV; if any exceptions occur during this process, we will catch them and log an error without crashing the entire script, allowing us to continue processing any remaining datasets
            process_dataset(path, args.outdir) # call the process_dataset function with the current dataset path and the output directory specified by the user (or the default "data" directory) to perform the end-to-end processing for that dataset; this will load the cleaned CSV, extract features, and save the resulting feature-rich CSV to the output directory
            success += 1 # if the process_dataset function completes without raising an exception, we increment the success counter to indicate that this dataset was processed successfully; this allows us to keep track of how many datasets were processed successfully and report it at the end
        except Exception as exc: # if any exception occurs during the processing of a dataset (such as file not found, parsing errors, etc.), we catch the exception and log an error message that includes the path of the dataset that failed and the exception message; we also log a debug message with the full traceback of the exception to help diagnose the issue; this allows us to handle errors gracefully without crashing the entire script, and to continue processing any remaining datasets in the list
            log.error("Failed to process %s: %s", path, exc) # log an error message that includes the path of the dataset that failed to process and the exception message, which helps identify which dataset had an issue and what the error was; this is important for diagnosing problems with specific datasets without crashing the entire script
            log.debug(traceback.format_exc()) # log a debug message with the full traceback of the exception, which provides detailed information about where and why the error occurred in the code; this is useful for developers to diagnose and fix issues with specific datasets without crashing the entire script

    log.info("-" * 60)
    log.info(
        "Done. %d / %d dataset(s) processed successfully.",
        success, len(args.datasets) # log the final summary of how many datasets were processed successfully out of the total number of datasets specified, which gives users a clear overview of the success rate of the processing and indicates if there were any issues with specific datasets that need to be addressed
    )

    if success < len(args.datasets): # if the number of successfully processed datasets is less than the total number of datasets specified, it means that some datasets failed to process; in this case, we log a warning to indicate that there were issues with some datasets and that users should check the logs for details on which datasets failed and why; we also exit with a non-zero status code to indicate that there were errors during processing, which can be useful for automated scripts or pipelines that check the exit status of this script
        sys.exit(1) # exit with a non-zero status code to indicate that there were errors during processing, which can be useful for automated scripts or pipelines that check the exit status of this script; this signals that not all datasets were processed successfully and that users should check the logs for details on which datasets failed and why


if __name__ == "__main__": # check if this script is being run directly (as the main program) rather than imported as a module; if it is being run directly, we call the main() function to execute the feature extraction process; this is a common Python idiom that allows the script to be used both as a standalone program and as an importable module without executing the main function when imported
    main() # call the main() function to execute the feature extraction process when this script is run directly, which will parse command-line arguments, process each specified dataset, and log the results; this serves as the entry point for the script when executed as a standalone program