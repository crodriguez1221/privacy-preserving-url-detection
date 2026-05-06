# --------------------------------------------------------------
# src/convert_phishtank.py
# --------------------------------------------------------------
# Purpose:
#   Convert the PhishTank online-valid.xml file into a clean
#   two-column CSV (url, label) ready for your project pipeline.
#
# Usage:
#   python src/convert_phishtank.py
#   python src/convert_phishtank.py --input data/online-valid.xml
# --------------------------------------------------------------

import os # for file path handling
import sys # for sys.exit if something goes wrong
import logging # for logging progress and errors
import argparse # for passing options via command-line (argument parsing)
import traceback # for detailed error tracebacks in logs
import xml.etree.ElementTree as ET # for parsing XML files

import pandas as pd # for data manipulation and saving spreadsheets style data (CSV files)


logging.basicConfig( # configure logging to show timestamps and log levels
    level=logging.INFO, # show INFO and above (INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s", # include timestamp, log level, and message
    datefmt="%H:%M:%S", # show only time in logs for readability
)
log = logging.getLogger(__name__) # create a logger for this module (src/convert_phishtank.py)

# --------------------------------------------------------------
# Default file paths
# --------------------------------------------------------------
DEFAULT_INPUT  = os.path.join("data", "online-valid.xml") # filename for PhishTank XML download
DEFAULT_OUTPUT = os.path.join("data", "phishtank_urls.csv") # filename for output CSV (will be created if it doesn't exist)

# --------------------------------------------------------------
# XML Conversion function
# Note: PhishTank XML structure can vary, so we try multiple tags to find the URL.
# --------------------------------------------------------------
def convert_phishtank_xml(input_path: str, output_path: str) -> None: # main function to convert PhishTank XML to CSV
    """
    Read PhishTank XML and write a clean url + label CSV.
    We extract only the <url> field and label everything 1 (phishing).
    """
    if not os.path.exists(input_path): # check if the input XML file exists before trying to read it
        log.error("File not found: %s", input_path) # if the file doesn't exist, log an error and exit
        log.error("Make sure online-valid.xml is in your data\\ folder.") # common mistake is to forget to download the file or put it in the wrong folder
        sys.exit(1) # exit with error code 1 to indicate failure

    log.info("Loading PhishTank XML from %s", input_path) # log that we're starting to load the XML file
    log.info("This may take a moment for large files...") # warn the user that loading may take time, especially if the XML file is large

    try:
        tree = ET.parse(input_path) # try to parse the XML file into an ElementTree object
        root = tree.getroot() # get the root element of the XML tree, which is the starting point for navigating the XML structure
    except ET.ParseError as e:  # catch parsing errors that occur if the XML file is malformed or corrupted
        log.error("Could not parse XML file: %s", e) # if there's a parsing error (e.g. malformed XML), log the error message and exit
        log.error("The file may be corrupted. Try downloading it again.") # common issue is that the file was not downloaded correctly or is incomplete, leading to parsing errors
        sys.exit(1) # exit with error code 1 to indicate failure

    log.info("XML loaded successfully. Extracting URLs...") # log that we've successfully loaded the XML and are now starting to extract URLs from it

    records = [] # this will hold the extracted URL records as dictionaries with 'url' and 'label' keys

    for entry in root.iter("entry"): # PhishTank XML uses <entry> tage, iterate over all <entry> elements in the XML tree, regardless of their nesting level, to ensure we capture all entries even if the XML structure varies

        # Extracting URL 
        url_el = entry.find("url") # first try to find the URL in the <url> tag
        if url_el is None: # if <url> tag is not found, try the alternative <phish_url> tag which is also commonly used in PhishTank XML
            url_el = entry.find("phish_url") # if <phish_url> is also not found, url_el will remain None
        if url_el is None or not url_el.text: # if we still don't have a URL element or if the URL element is empty, skip this entry and move to the next one
            continue # if there's no URL found in either tag, we can't use this entry, so we skip it

        url = url_el.text.strip() # if we found a URL element, get its text content and strip any leading/trailing whitespace to clean it up
        if not url: # if the URL is empty after stripping whitespace, skip this entry as well since it doesn't contain a valid URL
            continue # if the URL is empty, we can't use this entry, so we skip it

        # Only keep verified entries for data quality
        verified_el = entry.find("verified") # check if there's a <verified> tag in the entry, which indicates whether the phishing URL has been verified by PhishTank
        if verified_el is not None: # if the <verified> tag exists, we want to check its value to ensure we're only keeping verified phishing URLs for better data quality
            if str(verified_el.text).strip().lower() != "yes": # if the <verified> tag's text is not "yes" (case-insensitive), it means this entry is not verified, so we skip it to maintain data quality
                continue # if the entry is not verified, we skip it to ensure our dataset only contains high-quality, verified phishing URLs

        records.append({"url": url, "label": 1}) # if we have a valid URL and it's verified (or if there's no verification info), we add it to our records list with a label of 1 indicating it's a phishing URL

    log.info("Extracted %d entries from XML.", len(records)) # log how many entries we extracted after filtering for verified URLs

    # If verified filter removed everything, keep all entries
    if not records: # if we ended up with no records after filtering for verified entries, it means that either all entries were unverified or the XML structure is different than expected, so we log a warning and try to extract URLs without filtering for verification to avoid ending up with an empty dataset
        log.warning(
            "No verified entries found - keeping all entries without filtering."
        )
        for entry in root.iter("entry"): # if we have no verified entries, we iterate over all <entry> elements again to extract URLs without checking for verification
            url_el = entry.find("url") or entry.find("phish_url") # try to find the URL in either <url> or <phish_url> tags, similar to before, but this time we won't check for verification
            if url_el is not None and url_el.text: # if we find a URL element and it has text, we add it to our records list with a label of 1, even if it's not verified, to ensure we have some data to work with
                records.append({"url": url_el.text.strip(), "label": 1}) # if we find a URL, we add it to our records list with a label of 1, even if it's not verified, to ensure we have some data to work with

    if not records: # if we still have no records after trying to extract URLs without verification filtering, it means that the XML structure may be different than expected or there are no valid URL entries, so we log an error and exit to avoid creating an empty dataset
        log.error(
            "No URLs could be extracted. The XML structure may be different "
            "than expected. Check the file manually in Notepad."
        )
        sys.exit(1) # exit with error code 1 to indicate failure if we couldn't extract any URLs, which likely means the XML structure is different than expected or there are no valid URL entries

    df = pd.DataFrame(records) # convert the list of records (dictionaries) into a pandas DataFrame for easier manipulation and saving to CSV

    # Remove duplicates
    before = len(df) # store the number of records before removing duplicates so we can log how many duplicates were removed
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True) # remove duplicate URLs from the DataFrame to ensure we have unique URLs in our dataset, and reset the index after dropping duplicates for cleanliness
    log.info(
        "Removed %d duplicate URLs. Final count: %d",
        before - len(df), len(df) # log how many duplicates were removed and how many unique URLs remain in the final dataset
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True) # ensure the output directory exists before trying to save the CSV file, creating it if necessary, and handle the case where output_path is just a filename without a directory
    df.to_csv(output_path, index=False) # save the DataFrame to a CSV file at the specified output path without including the index column, which is not needed for our dataset

    log.info("Saved -> %s", os.path.abspath(output_path)) # log the absolute path of the saved CSV file for user reference and confirmation that the file was created successfully
    log.info("Columns: url, label (all values = 1 meaning phishing)") # log the columns of the output CSV and explain that all label values are 1, indicating that all URLs in this dataset are phishing URLs
    log.info(
        "Next step: python src/prepare_data.py --datasets %s", output_path # log the next step for the user, which is to run the prepare_data.py script with the output CSV file as input to continue building the dataset for training and evaluation
    )

# --------------------------------------------------------------
# CLI
# We use argparse to allow users to specify input and output file paths via command-line arguments, with defaults set to our expected file locations.
# --------------------------------------------------------------
def parse_args() -> argparse.Namespace: # function to parse command-line arguments for input and output file paths
    p = argparse.ArgumentParser( # create an ArgumentParser object to handle command-line arguments and provide help messages
        description="Convert PhishTank XML to a clean CSV." # description of what this script does, shown when the user runs the script with --help
    )
    p.add_argument( # add an argument for the input file path, allowing the user to specify a custom path or use the default
        "--input", "-i", # allow the user to specify the input file path using --input or -i flags
        default=DEFAULT_INPUT, # set the default input path to our expected location for the PhishTank XML file, so users can simply run the script without arguments if they have the file in the right place
        help=f"Path to PhishTank XML file (default: {DEFAULT_INPUT})", # help message for the input argument, showing the default value for user reference
    )
    p.add_argument( # add an argument for the output file path, allowing the user to specify a custom path or use the default
        "--output", "-o", # allow the user to specify the output file path using --output or -o flags
        default=DEFAULT_OUTPUT, # set the default output path to our expected location for the output CSV file, so users can simply run the script without arguments if they want the output in the default location
        help=f"Path to save output CSV (default: {DEFAULT_OUTPUT})", # help message for the output argument, showing the default value for user reference
    )
    return p.parse_args() # parse the command-line arguments and return them as a Namespace object for use in the main function

# --------------------------------------------------------------
# Entry point
# The main function serves as the entry point for the script, calling the argument parser and the conversion function, and handling any unexpected errors gracefully with logging.
# --------------------------------------------------------------
def main() -> None: # main function that serves as the entry point for the script
    args = parse_args() # parse command-line arguments to get input and output file paths
    try:
        convert_phishtank_xml(args.input, args.output) # call the conversion function with the specified input and output paths, and wrap it in a try-except block to catch any unexpected errors that may occur during the conversion process
        log.info("Conversion complete.") # log that the conversion process is complete if no exceptions were raised
    except Exception:
        log.error("Unhandled error:\n%s", traceback.format_exc()) # if any unexpected exception occurs during the conversion process, log the error with a full traceback for debugging purposes
        sys.exit(1) # exit with error code 1 to indicate failure if an unhandled exception occurred during the conversion process


if __name__ == "__main__": # standard Python idiom to check if this script is being run directly (as the main program) rather than imported as a module, and if so, call the main function to execute the script's functionality
    main() # call the main function to execute the script when run directly, which will handle argument parsing and call the conversion function, while also catching and logging any unexpected errors that may occur during execution