# -*- coding: utf-8 -*-
# --------------------------------------------------------------
# src/convert_urlhaus.py
# --------------------------------------------------------------
# Purpose:
#   Convert the URLhaus abuse.ch raw download (.txt) into a
#   clean two-column CSV (url, label) ready for build_dataset.py.
#
#   This script mirrors convert_phishtank.py in purpose:
#   each converter handles ONE raw source file and produces
#   ONE standardised CSV.
#
# URLhaus file format:
#   - Lines beginning with # are comments and are skipped
#   - Remaining lines are CSV with 9 columns:
#       id, dateadded, url, url_status, last_online,
#       threat, tags, urlhaus_link, reporter
#   - All URLs are labelled malicious (label = 1)
#   - Both online and offline URLs are retained
#     (offline URLs were verified malicious when active;
#      excluding them would discard the majority of the dataset)
#
# Input:
#   data/urlhaus_abuse_ch.txt
#
# Output:
#   data/urlhaus_urls.csv   (columns: url, label)
#
# Usage:
#   python src/convert_urlhaus.py
#   python src/convert_urlhaus.py --input data/urlhaus_abuse_ch.txt
#   python src/convert_urlhaus.py --input data/urlhaus_abuse_ch.txt --output data/urlhaus_urls.csv
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
log = logging.getLogger(__name__) # create a logger for this module (src/convert_urlhaus.py)

# --------------------------------------------------------------
# Default paths
# --------------------------------------------------------------
DEFAULT_INPUT  = os.path.join("data", "urlhaus_abuse_ch.txt") # download file from https://urlhaus.abuse.ch/downloads/csv_recent/
DEFAULT_OUTPUT = os.path.join("data", "urlhaus_urls.csv") # output CSV with columns: url, label

# Expected column names after skipping comment lines
URLHAUS_COLUMNS = [
    "id", "dateadded", "url", "url_status", "last_online",
    "threat", "tags", "urlhaus_link", "reporter",
]


# --------------------------------------------------------------
# Converter
# --------------------------------------------------------------
def convert(input_path: str, output_path: str) -> None: # main function to convert URLhaus .txt to CSV
    """
    Read the URLhaus .txt file and save a clean url + label CSV.
    We extract the 'url' column, label everything 1 (phishing).
    """
    if not os.path.exists(input_path): # check if the input file exists before trying to read it
        log.error( # log an error if the file is missing, and suggest where to get it
            "Input file not found: %s\n"
            "  Download from https://urlhaus.abuse.ch/downloads/csv_recent/\n"
            "  and save as %s",
            input_path, input_path # suggest where to get the file and what path to save it as
        )
        sys.exit(1) # exit with error code if input file is missing

    log.info("Reading URLhaus file: %s", os.path.abspath(input_path)) # log the absolute path of the input file being read for transparency

    # Read the file, skipping lines that start with #
    try:
        df = pd.read_csv( # read the CSV file using pandas, with specified options
            input_path, # path to the input file
            comment="#", # skip lines starting with # as comments
            header=None, # no header row in the file, so treat all lines as data
            names=URLHAUS_COLUMNS, # assign column names based on expected format
            on_bad_lines="skip", # skip lines that don't match the expected format instead of raising an error
            encoding="utf-8", # try reading with utf-8 encoding first, which is common for text files
        )
    except UnicodeDecodeError: # if utf-8 decoding fails, try latin-1 which can handle a wider range of byte sequences
        df = pd.read_csv( # read the file again with latin-1 encoding as a fallback if utf-8 fails
            input_path, # same file path
            comment="#", # same options for skipping comments and handling bad lines
            header=None, # same as before, no header row
            names=URLHAUS_COLUMNS, # same column names
            on_bad_lines="skip", # same option to skip bad lines
            encoding="latin-1", # fallback encoding that can handle more byte sequences, but may not be correct for all characters
        )
        log.warning("Used latin-1 encoding fallback.") # log a warning if we had to use the fallback encoding, which may indicate potential issues with character encoding in the input file

    log.info("Raw rows loaded: %d", len(df)) # log the number of rows loaded from the file before any processing for transparency

    # Log URL status breakdown for transparency
    if "url_status" in df.columns: # check if the url_status column is present before trying to analyze it
        status_counts = df["url_status"].value_counts(dropna=False) # count the occurrences of each unique value in the url_status column, including not a number (NaN) values
        log.info("URL status breakdown:") # log the breakdown of URL statuses for transparency, which can help understand the composition of the dataset (e.g., how many are online vs offline)
        for status, count in status_counts.items(): # iterate through each unique status and its count, and log it in a formatted way
            log.info("  %-12s : %d", status, count) # log the status and count with formatting for alignment (status left-aligned in 12 characters, count as an integer)

    # Keep only the url column and drop any missing values
    df = df[["url"]].dropna().copy() # select only the 'url' column and drop rows where 'url' is missing (NaN), then make a copy to avoid SettingWithCopyWarning in pandas
    df["url"] = df["url"].astype(str).str.strip() # ensure all URLs are strings and remove leading/trailing whitespace for consistency
    df = df[df["url"] != ""] # remove any rows where the URL is an empty string after stripping whitespace, as these are not valid URLs

    # Remove duplicates
    before = len(df) # store the number of rows before removing duplicates for logging purposes
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True) # drop duplicate URLs based on the 'url' column and reset the index after dropping duplicates for a clean DataFrame
    log.info(
        "Removed %d duplicate URLs. Unique URLs remaining: %d",
        before - len(df), len(df) # log how many duplicates were removed and how many unique URLs remain after deduplication for transparency
    )

    df["label"] = 1 # add a new column 'label' and set it to 1 for all rows, indicating that all URLs in this dataset are malicious (phishing) based on the source (URLhaus)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True) # ensure the output directory exists before trying to save the file, creating it if necessary (os.path.dirname returns the directory part of the path, and os.makedirs creates it if it doesn't exist; the 'or "."' handles the case where output_path is just a filename without a directory)
    df.to_csv(output_path, index=False) # save the DataFrame to a CSV file at the specified output path, without including the index column in the output file for a clean CSV format

    log.info("-" * 60) # log a separator line for readability in the logs
    log.info("Conversion complete.") # log that the conversion process is complete for transparency
    log.info("  Input  : %s", os.path.abspath(input_path)) # log the absolute path of the input file for transparency, confirming what file was processed
    log.info("  Output : %s", os.path.abspath(output_path)) # log the absolute path of the output file for transparency, confirming where the results were saved
    log.info("  Rows   : %d malicious URLs (label=1)", len(df)) # log the number of rows in the final dataset, which corresponds to the number of unique malicious URLs extracted and saved to the output CSV for transparency
    log.info("-" * 60) # log another separator line for readability in the logs
    log.info("Next step: python src/build_dataset.py") # log a suggestion for the next step in the workflow, which is to run the build_dataset.py script to combine this converted dataset with others and prepare it for model training


# --------------------------------------------------------------
# CLI
# We use argparse to allow users to specify input and output file paths via command-line arguments, with defaults set to our expected file locations.
# --------------------------------------------------------------
def parse_args() -> argparse.Namespace: # function to parse command-line arguments for input and output file paths
    p = argparse.ArgumentParser( # create an ArgumentParser object to handle command-line arguments and provide help messages
        description=( # description of what this script does, shown when the user runs the script with --help
            "Convert URLhaus abuse.ch .txt download to a clean "
            "url + label CSV ready for build_dataset.py."
        )
    )
    p.add_argument( # add an argument for the input file path, allowing the user to specify a custom path or use the default
        "--input", "-i", # allow the user to specify the input file path using --input or -i flags
        default=DEFAULT_INPUT, # set the default input path to our expected location for the URLhaus text file, so users can simply run the script without arguments if they have the file in the right place
        help=f"Path to URLhaus .txt file (default: {DEFAULT_INPUT})", #  help message for the input argument, showing the default value for user reference
    )
    p.add_argument( # add an argument for the output file path, allowing the user to specify a custom path or use the default
        "--output", "-o", # allow the user to specify the output file path using --output or -o flags
        default=DEFAULT_OUTPUT, # set the default output path to our expected location for the output CSV file, so users can simply run the script without arguments if they want the output in the default location
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})", # help message for the output argument, showing the default value for user reference
    )
    return p.parse_args() # parse the command-line arguments and return them as a Namespace object for use in the main function

# --------------------------------------------------------------
# Entry point
# --------------------------------------------------------------
if __name__ == "__main__": # main function that servces as the entry point for the script
    try:
        args = parse_args() # parse command-line arguments to get input and output file paths
        convert(args.input, args.output) # call the conversion function with the specified input and output paths, and wrap it in a try-except block to catch any unexpected errors that may occur during the conversion process
    except Exception:
        log.error("Unhandled error:\n%s", traceback.format_exc()) # if any unexpected exception occurs during the conversion process, log the error with a full traceback for debugging purposes
        sys.exit(1) # exit with error code 1 to indicate failure if an unhandled exception occurred during the conversion process
