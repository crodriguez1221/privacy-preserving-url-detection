# -*- coding: utf-8 -*-
# --------------------------------------------------------------
# src/build_dataset.py
# --------------------------------------------------------------
# Purpose:
#   Build two separate balanced datasets for phishing detection
#   by pairing each malicious source with an equal number of
#   legitimate URLs from Tranco.
#
#   Dataset 1: phishtank_tranco_dataset.csv
#       PhishTank phishing URLs  (label=1)
#       + matching Tranco legitimate URLs (label=0)
#
#   Dataset 2: urlhaus_tranco_dataset.csv
#       URLhaus malicious URLs   (label=1)
#       + matching Tranco legitimate URLs (label=0)
#
#   Tranco is read independently for each dataset so both
#   receive their own fresh set of legitimate URLs.
#
#   Each malicious source is capped at DEFAULT_LIMIT URLs.
#   The same number of Tranco URLs is taken to ensure a
#   perfectly balanced 50/50 class split in both datasets.
#
# Input files (each malicious file produced by their own converter):
#   data/phishtank_urls.csv    <- convert_phishtank.py    <- online-valid.xml
#   data/urlhaus_urls.csv      <- convert_urlhaus.py      <- urlhaus_abuse.ch.txt
#   data/tranco_top_sites.csv  <- downloaded from tranco-list.eu
#
# Output files:
#   data/phishtank_tranco_dataset.csv
#   data/urlhaus_tranco_dataset.csv
#
# Usage:
#   python src/build_dataset.py
#   python src/build_dataset.py --limit 10000
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
log = logging.getLogger(__name__) # create a logger for this module (src/build_dataset.py)

# --------------------------------------------------------------
# Default file paths
# Where the files are and were they are saved by default if no command-line arguments are given.
# The malicious CSV files are expected to be produced by their respective converter scripts, which should be run first to generate these input files.
# --------------------------------------------------------------
DEFAULT_PHISHTANK   = os.path.join("data", "phishtank_urls.csv") # expected to be produced by convert_phishtank.py from the online-valid.xml file downloaded from PhishTank
DEFAULT_URLHAUS     = os.path.join("data", "urlhaus_urls.csv") # expected to be produced by convert_urlhaus.py from the urlhaus_abuse.ch.txt file downloaded from URLhaus
DEFAULT_TRANCO      = os.path.join("data", "tranco_top_sites.csv") # expected to be downloaded from tranco-list.eu and saved as specified in the help message (should contain rank and domain columns without a header row)
DEFAULT_OUT_PHISH   = os.path.join("data", "phishtank_tranco_dataset.csv") # the output file for the PhishTank + Tranco dataset, which will be saved as a CSV file containing the combined and shuffled dataset of phishing URLs from PhishTank and legitimate URLs from Tranco
DEFAULT_OUT_URLHAUS = os.path.join("data", "urlhaus_tranco_dataset.csv") # the output file for the URLhaus + Tranco dataset, which will be saved as a CSV file containing the combined and shuffled dataset of malicious URLs from URLhaus and legitimate URLs from Tranco

# Max malicious URLs per dataset before balancing.
# Legitimate URLs are matched to this number automatically.
DEFAULT_LIMIT = 10000


# --------------------------------------------------------------
# Loaders
# Each loader reads a specific input file, checks for required columns, and returns a DataFrame with 'url' and 'label' columns.
# Each loader also logs how many URLs were loaded and any issues encountered.
# Each loader exits the program with an error message if the expected file is missing or malformed, since we can't proceed without the data.
# Each loader also ensures that the 'url' column is cleaned of duplicates and missing values, and that the 'label' column is set to 1 for malicious sources and 0 for legitimate sources.
# --------------------------------------------------------------
def load_phishtank(path: str) -> pd.DataFrame: # load the PhishTank CSV file and return a DataFrame with columns 'url' and 'label'
    """Load the PhishTank CSV produced by convert_phishtank.py."""
    if not os.path.exists(path): # check if the file exists at the given path
        log.error("PhishTank file not found: %s", path) # log an error if the file is missing
        log.error("Run convert_phishtank.py first.") # log a hint to the user about how to generate the missing file using the converter script
        sys.exit(1) # exit the program with an error code since we can't proceed without the data

    df = pd.read_csv(path, low_memory=False) # read the CSV file into a DataFrame, using low_memory=False to avoid dtype (data type object) issues with large files

    if "url" not in df.columns: # check if the expected 'url' column is present in the DataFrame
        log.error( # log an error if the 'url' column is missing, showing the actual columns found for debugging
            "No 'url' column in %s. Columns found: %s", # log the path of the file and the list of columns that were actually found in the DataFrame
            path, list(df.columns) # convert the columns to a list for easier readability in the logs
        )
        sys.exit(1) # exit the program with an error code since we can't proceed without the expected 'url' column

    df = df[["url"]].dropna().drop_duplicates() # keep only the 'url' column, drop rows with missing values in 'url', and remove duplicate URLs to ensure clean data
    df["label"] = 1 # add a 'label' column and set it to 1 for all rows since these are malicious URLs from PhishTank
    log.info("PhishTank  : loaded %d phishing URLs", len(df)) # log how many phishing URLs were loaded from the PhishTank CSV for transparency and debugging
    return df.reset_index(drop=True) # reset the index of the DataFrame after dropping duplicates and return it for further processing 


def load_urlhaus(path: str) -> pd.DataFrame: # load the URLhaus CSV file and return a DataFrame with columns 'url' and 'label'
    """
    Load the URLhaus CSV produced by convert_urlhaus.py.
    Expected columns: url, label (all 1s for malicious).
    Run convert_urlhaus.py first to generate this file.
    """
    if not os.path.exists(path): # check if the file exists at the given path
        log.error("URLhaus CSV not found: %s", path) # log an error if the file is missing, showing the expected path for debugging
        log.error("Run convert_urlhaus.py first to generate this file.") # log a hint to the user about how to generate the missing file using the converter script
        sys.exit(1) # exit the program with an error code since we can't proceed without the data

    df = pd.read_csv(path, low_memory=False) # read the CSV file into a DataFrame, using low_memory=False to avoid dtype (data type object) issues with large files

    if "url" not in df.columns: # check if the expected 'url' column is present in the DataFrame
        log.error( # log an error if the 'url' column is missing, showing the actual columns found for debugging
            "No 'url' column in %s. Columns found: %s", # log the path of the file and the list of columns that were actually found in the DataFrame
            path, list(df.columns) # convert the columns to a list for easier readability in the logs
        )
        sys.exit(1) # exit the program with an error code since we can't proceed without the expected 'url' column

    df = df[["url"]].dropna().drop_duplicates() # keep only the 'url' column, drop rows with missing values in 'url', and remove duplicate URLs to ensure clean data
    df["label"] = 1 # add a 'label' column and set it to 1 for all rows since these are malicious URLs from URLhaus

    log.info("URLhaus    : loaded %d malicious URLs", len(df)) # log how many malicious URLs were loaded from the URLhaus CSV for transparency and debugging
    return df.reset_index(drop=True) # reset the index of the DataFrame after dropping duplicates and return it for further processing


def load_tranco(path: str, n: int) -> pd.DataFrame: # load the Tranco CSV file, take the top n rows, and return a DataFrame with columns 'url' and 'label' (0 for legitimate)
    """
    Load the top n legitimate domains from Tranco.
    Tranco CSV format: rank, domain (no header row).
    Prepends https:// to each domain to form a full URL.
    """
    if not os.path.exists(path): # check if the file exists at the given path
        log.error("Tranco file not found: %s", path) # log an error if the file is missing, showing the expected path for debugging
        log.error(
            "Download from tranco-list.eu and save as data\\tranco_top_sites.csv" # log a hint to the user about how to obtain the missing Tranco file, including the expected URL and save location
        )
        sys.exit(1) # exit the program with an error code since we can't proceed without the data

    try:
        df = pd.read_csv( # read the Tranco CSV file into a DataFrame
            path, header=None, names=["rank", "domain"] # specify that there is no header row and name the columns 'rank' and 'domain' for easier processing later
        )
    except Exception as e: # catch any exceptions that occur while reading the CSV file, such as parsing errors or encoding issues
        log.error("Could not read Tranco file: %s", e) # log an error with the exception message to help diagnose what went wrong when trying to read the Tranco file
        sys.exit(1) # exit the program with an error code since we can't proceed without successfully reading the Tranco data

    df = df.head(n).copy() # take only the top n rows from the Tranco DataFrame to match the number of malicious URLs we need for balancing, and create a copy to avoid SettingWithCopyWarning when modifying the DataFrame later
    df["url"]   = "https://" + df["domain"].astype(str).str.strip() # create a new 'url' column by prepending 'https://' to the 'domain' column, ensuring that we have full URLs for the legitimate sites; also convert to string and strip whitespace just in case there are any formatting issues in the domain names
    df["label"] = 0 # add a 'label' column and set it to 0 for all rows since these are legitimate URLs from Tranco

    log.info("Tranco     : loaded %d legitimate URLs", len(df)) # log how many legitimate URLs were loaded from the Tranco CSV for transparency and debugging
    return df[["url", "label"]].reset_index(drop=True) # return only the 'url' and 'label' columns, reset the index of the DataFrame, and return it for further processing


# --------------------------------------------------------------
# Build one balanced dataset
# This function takes a DataFrame of malicious URLs, loads an equal number of legitimate URLs from Tranco, combines them into one DataFrame, removes any duplicates across sources, shuffles the rows, and saves the final balanced dataset to a CSV file.
# The function also logs the progress and summary statistics of the dataset being built, including how many malicious and legitimate URLs were included and any issues encountered during the process.
# The function is designed to be reusable for both the PhishTank and URLhaus datasets by passing in the appropriate DataFrame and output path.
# --------------------------------------------------------------
def build_one_dataset( # build a balanced dataset by combining one malicious source with matching Tranco legitimate URLs, then shuffle and save to a CSV file
    phishing_df: pd.DataFrame, # the DataFrame containing malicious URLs (with 'url' and 'label' columns) that we want to combine with legitimate URLs from Tranco
    tranco_path: str, # the file path to the Tranco CSV file, which will be loaded to get legitimate URLs for balancing the dataset
    output_path: str, # the file path where the final combined dataset will be saved as a CSV file after processing
    source_name: str, # a name for the dataset being built (e.g., "PhishTank + Tranco") used for logging purposes to identify which dataset is being processed
    limit: int, # an optional limit on the number of malicious URLs to use from the phishing_df; if provided, the phishing_df will be sampled down to this number to ensure that we don't exceed the desired dataset size and that the dataset remains balanced with an equal number of legitimate URLs from Tranco
) -> None:
    """
    Combine one malicious source with matching Tranco legitimate URLs,
    creating a perfectly balanced 50/50 dataset, then shuffle and save.
    """
    log.info("=" * 60) # log a separator line for better readability in the logs
    log.info("Building dataset: %s", source_name) # log the name of the dataset being built for clarity in the logs
    log.info("=" * 60) # log another separator line for better readability in the logs

    # Cap malicious URLs at limit
    if limit and len(phishing_df) > limit: # check if a limit is set and if the number of malicious URLs exceeds that limit
        phishing_df = phishing_df.sample( # randomly sample down to the specified limit to ensure we don't exceed the desired dataset size and that the dataset remains balanced with an equal number of legitimate URLs from Tranco; also reset the index after sampling
            n=limit, random_state=42 # use a fixed random state for reproducibility so that the same sample is taken each time the script is run with the same limit, which is important for consistent results and debugging (42 Answer to the Ultimate Question of Life Hitchhiker's Guide to the Galaxy :) )
        ).reset_index(drop=True) # reset the index of the DataFrame after sampling to ensure a clean index for further processing
        log.info("Malicious URLs capped at: %d", len(phishing_df)) # log the number of malicious URLs after applying the limit for transparency and debugging

    n_malicious = len(phishing_df) # store the number of malicious URLs that we will be using in this dataset, which is important for loading the correct number of legitimate URLs from Tranco to maintain a balanced dataset

    tranco_df = load_tranco(tranco_path, n_malicious) # load the Tranco CSV file and get exactly n_malicious legitimate URLs to match the number of malicious URLs we have (loads exactly as many), ensuring a perfectly balanced dataset

    if len(tranco_df) < n_malicious: # check if we were able to load enough legitimate URLs from Tranco to match the number of malicious URLs; if not, log a warning that the dataset will not be perfectly balanced and show how many rows were loaded versus how many were needed
        log.warning(
            "Tranco only provided %d rows but %d were needed -- "
            "dataset will not be perfectly balanced.",
            len(tranco_df), n_malicious # log the number of legitimate URLs loaded from Tranco and the number of malicious URLs we needed to match for transparency and debugging
        )

    combined = pd.concat(
        [phishing_df, tranco_df], ignore_index=True # combine the malicious and legitimate DataFrames into one DataFrame, ignoring the original indices to create a new continuous index for the combined dataset
    )

    # Remove any accidental cross-source duplicates
    before = len(combined) # store the number of rows in the combined DataFrame before removing duplicates for logging purposes to show how many duplicates were removed across sources
    combined = combined.drop_duplicates( # remove any duplicate URLs (below)
        subset=["url"] # specify that we want to drop duplicates based on the 'url' column, so if the same URL appears more than once (regardless of label), it will be removed to ensure that each URL is unique in the dataset
    ).reset_index(drop=True) # reset the index of the DataFrame after dropping duplicates to ensure a clean index for further processing
    if before - len(combined) > 0: # check if any duplicates were removed by comparing the number of rows before and after dropping duplicates
        log.info(
            "Removed %d cross-source duplicate URLs.",
            before - len(combined) # log the number of duplicate URLs that were removed across sources for transparency and debugging, showing how many duplicates were found and removed to ensure a clean dataset
        )

    # Shuffle rows
    combined = combined.sample( 
        frac=1, random_state=42 # shuffle the rows of the combined DataFrame randomly to ensure that the malicious and legitimate URLs are mixed together, which is important for training machine learning models that expect a random distribution of classes
    ).reset_index(drop=True) # reset the index of the DataFrame after shuffling to ensure a clean index for the final dataset

    # Summary
    n_mal   = int((combined["label"] == 1).sum()) # count how many malicious URLs are in the combined dataset by summing the 'label' column where label=1 indicates malicious URLs
    n_legit = int((combined["label"] == 0).sum()) # count how many legitimate URLs are in the combined dataset by summing the 'label' column where label=0 indicates legitimate URLs
    pct     = 100 * n_mal / len(combined) if len(combined) > 0 else 0 # calculate the percentage of malicious URLs in the combined dataset for logging purposes, ensuring that we don't divide by zero if the combined dataset is empty (though it shouldn't be if everything is working correctly)

    log.info("-" * 60) # log a separator line for better readability in the logs
    log.info("Dataset : %s", source_name) # log the name of the dataset for clarity in the logs
    log.info("  Total rows  : %d", len(combined)) # log the total number of rows in the combined dataset for transparency and debugging
    log.info("  Malicious   : %d  (%.1f%%)", n_mal, pct) # log the number and percentage of malicious URLs in the combined dataset for transparency and debugging
    log.info("  Legitimate  : %d  (%.1f%%)", n_legit, 100 - pct) # log the number and percentage of legitimate URLs in the combined dataset for transparency and debugging
    log.info("-" * 60) # log another separator line for better readability in the logs

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True) # ensure that the directory for the output file exists, creating it if necessary; os.path.dirname(output_path) gets the directory part of the output path, and if it's empty (e.g., just a filename), we use "." to refer to the current directory; exist_ok=True means that it won't raise an error if the directory already exists, which is useful for idempotent script runs
    combined.to_csv(output_path, index=False) # save the combined DataFrame to a CSV file at the specified output path, without including the index in the CSV file since we don't need it for our dataset
    log.info("Saved -> %s", os.path.abspath(output_path)) # log the absolute path of the saved CSV file for confirmation and debugging, showing where the final dataset has been saved on the filesystem


# --------------------------------------------------------------
# CLI
# This function uses argparse to define command-line arguments for the script, allowing users to specify custom file paths for the input CSV files (PhishTank, URLhaus, Tranco) and the output dataset files, as well as a limit on the number of malicious URLs to include in each dataset.
# The function also provides default values for all arguments, so the script can be run without any arguments to use the default file paths and limit, but also allows for flexibility if the user wants to specify different files or limits.
# The function returns a Namespace object containing the parsed arguments, which can be used in the main function to access the specified file paths and limit for building the datasets.
# The CLI also includes help messages for each argument, which can be accessed by running the script with the --help flag, providing users with information about what each argument does and what the expected input is.
# --------------------------------------------------------------
def parse_args() -> argparse.Namespace: # define command-line arguments for the script using argparse, allowing users to specify custom file paths and limits for building the datasets
    p = argparse.ArgumentParser( # create an ArgumentParser object to handle command-line arguments, which will automatically generate help messages and parse the arguments when the script is run
        description=( 
            "Build two balanced phishing detection datasets "
            "from PhishTank, URLhaus, and Tranco."
        )
    )
    p.add_argument( # define a command-line argument for the PhishTank CSV file path, with a default value and a help message explaining what it is and how it is produced
        "--phishtank",
        default=DEFAULT_PHISHTANK, # use the default file path for the PhishTank CSV if no argument is provided, which is expected to be produced by the convert_phishtank.py script
        help=f"PhishTank CSV (default: {DEFAULT_PHISHTANK})", # provide a help message that indicates this is the PhishTank CSV file and shows the default path for clarity in the help output
    )
    p.add_argument( # define a command-line argument for the URLhaus CSV file path, with a default value and a help message explaining what it is and how it is produced
        "--urlhaus",
        default=DEFAULT_URLHAUS, # use the default file path for the URLhaus CSV if no argument is provided, which is expected to be produced by the convert_urlhaus.py script
        help=f"URLhaus CSV from convert_urlhaus.py (default: {DEFAULT_URLHAUS})", # provide a help message that indicates this is the URLhaus CSV file, mentions that it should be produced by the convert_urlhaus.py script, and shows the default path for clarity in the help output
    )
    p.add_argument( # define a command-line argument for the Tranco CSV file path, with a default value and a help message explaining what it is and how to obtain it
        "--tranco",
        default=DEFAULT_TRANCO, # use the default file path for the Tranco CSV if no argument is provided, which is expected to be downloaded from tranco-list.eu and saved as specified in the help message
        help=f"Tranco CSV (default: {DEFAULT_TRANCO})", # provide a help message that indicates this is the Tranco CSV file, mentions that it should be downloaded from tranco-list.eu, and shows the default path for clarity in the help output
    )
    p.add_argument( # define a command-line argument for the output file path for the PhishTank + Tranco dataset, with a default value and a help message explaining what it is
        "--out-phishtank",
        default=DEFAULT_OUT_PHISH,# use the default file path for the output PhishTank + Tranco dataset if no argument is provided, which will be saved as specified in the default value
        help=f"Output for PhishTank dataset (default: {DEFAULT_OUT_PHISH})", # provide a help message that indicates this is the output file for the PhishTank + Tranco dataset and shows the default path for clarity in the help output
    )
    p.add_argument( # define a command-line argument for the output file path for the URLhaus + Tranco dataset, with a default value and a help message explaining what it is
        "--out-urlhaus",
        default=DEFAULT_OUT_URLHAUS, # use the default file path for the output URLhaus + Tranco dataset if no argument is provided, which will be saved as specified in the default value
        help=f"Output for URLhaus dataset (default: {DEFAULT_OUT_URLHAUS})", # provide a help message that indicates this is the output file for the URLhaus + Tranco dataset and shows the default path for clarity in the help output
    ) 
    p.add_argument( # define a command-line argument for the limit on the number of malicious URLs to include in each dataset, with a default value and a help message explaining what it does and how it affects the dataset
        "--limit",
        type=int, # specify that this argument should be an integer, which will be used to limit the number of malicious URLs included in each dataset to ensure that we don't exceed the desired dataset size and that the dataset remains balanced with an equal number of legitimate URLs from Tranco
        default=DEFAULT_LIMIT, # use the default limit for the number of malicious URLs if no argument is provided, which is set to 10000 to create a reasonably sized dataset while still being manageable for training machine learning models
        help=(
            "Max malicious URLs per dataset (default: 10000). "
            "Legitimate URLs are matched automatically."
        ),
    )
    return p.parse_args() # parse the command-line arguments and return them as a Namespace object for use in the main function


# --------------------------------------------------------------
# Entry point
# The main function is the entry point of the script, which orchestrates the loading of the input CSV files, building of the two balanced datasets, and saving of the final CSV files. 
# It also includes error handling to catch any unhandled exceptions and log them with a detailed traceback for debugging purposes.
# --------------------------------------------------------------
def main() -> None: 
    args = parse_args() # parse the command-line arguments to get the file paths and limit for building the datasets, which allows for flexibility in specifying different input files and limits when running the script

    try:
        phishtank_df = load_phishtank(args.phishtank) # load the PhishTank CSV file into a DataFrame, which will be used as the malicious source for the first dataset; this function will also log how many phishing URLs were loaded and any issues encountered during loading
        urlhaus_df   = load_urlhaus(args.urlhaus) # load the URLhaus CSV file into a DataFrame, which will be used as the malicious source for the second dataset; this function will also log how many malicious URLs were loaded and any issues encountered during loading

        # Dataset 1: PhishTank + Tranco
        # This will build the first dataset by combining the PhishTank malicious URLs with an equal number of legitimate URLs from Tranco, shuffling the combined dataset, and saving it to a CSV file. The function will also log the progress and summary statistics of the dataset being built.
        build_one_dataset(
            phishing_df=phishtank_df,
            tranco_path=args.tranco,
            output_path=args.out_phishtank,
            source_name="PhishTank + Tranco",
            limit=args.limit,
        )

        # Dataset 2: URLhaus + Tranco
        # This will build the second dataset by combining the URLhaus malicious URLs with an equal number of legitimate URLs from Tranco, shuffling the combined dataset, and saving it to a CSV file. The function will also log the progress and summary statistics of the dataset being built.
        build_one_dataset(
            phishing_df=urlhaus_df,
            tranco_path=args.tranco,
            output_path=args.out_urlhaus,
            source_name="URLhaus + Tranco",
            limit=args.limit,
        )

        log.info("=" * 60) # log a separator line for better readability in the logs
        log.info("Both datasets built successfully.") # log a message indicating that both datasets were built successfully for confirmation
        log.info("Next steps:") # log the next steps for the user, which is to run the prepare_data.py script on the generated datasets to prepare them for training machine learning models
        log.info(
            "  python src/prepare_data.py --datasets %s",
            args.out_phishtank # log the command to run the prepare_data.py script on the PhishTank + Tranco dataset, showing the output path of the generated dataset for clarity in the logs
        )
        log.info(
            "  python src/prepare_data.py --datasets %s",
            args.out_urlhaus # log the command to run the prepare_data.py script on the URLhaus + Tranco dataset, showing the output path of the generated dataset for clarity in the logs
        )
        log.info("=" * 60) # log another separator line for better readability in the logs

    except Exception:
        log.error("Unhandled error:\n%s", traceback.format_exc()) # catch any unhandled exceptions that occur during the execution of the main function, and log an error with the detailed traceback to help diagnose what went wrong
        sys.exit(1) # exit the program with an error code since an unhandled exception occurred, indicating that something went wrong during the execution of the script


if __name__ == "__main__":
    main() # call the main function when the script is run directly, which will execute the entire process of building the datasets as defined in the main function; this is a common Python idiom to allow for modular code and to prevent the main function from being executed if this script is imported as a module in another script.
