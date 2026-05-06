
# -*- coding: utf-8 -*-
# --------------------------------------------------------------
# src/predict.py
# --------------------------------------------------------------
# Purpose:
#   Classify a single user-supplied URL as Malicious or Legitimate
#   using the trained model artifacts saved by train_models.py.
#
#   This script does ONE job: inference.
#
#   It does NOT retrain any model. It loads the previously saved
#   .pkl files from outputs/ and applies the same 24-feature
#   engineering logic as extract_features.py to the input URL,
#   then runs it through all three trained classifiers and
#   returns a plain-language verdict for each model plus a
#   majority-vote summary.
#
#   All feature extraction is performed on the URL string alone —
#   no DNS lookups, no HTTP requests, no network connections of
#   any kind are made. The URL is treated as a passive string
#   from which features are derived mathematically.
#
# Prerequisites:
#   train_models.py must have been run at least once so that
#   the following .pkl files exist in outputs/:
#       outputs/model_lr.pkl         (Logistic Regression + scaler)
#       outputs/model_rf.pkl         (Random Forest)
#       outputs/model_dt.pkl         (Decision Tree)
#       outputs/scaler.pkl           (StandardScaler fitted on training data)
#
# Usage:
#   python src/predict.py --url https://example.com
#   python src/predict.py --url http://192.168.1.1/payload.exe
#   python src/predict.py --url suspicious-site.xyz/login/verify
# --------------------------------------------------------------

import os                          # for file path handling and checking if files exist
import sys                         # for sys.exit if something goes wrong
import logging                     # for logging progress, results, and errors
import argparse                    # for passing the URL via command-line argument
import traceback                   # for detailed error tracebacks if something unexpected happens
import pickle                      # for loading the serialised .pkl model and scaler files saved by train_models.py
from urllib.parse import urlparse  # for decomposing the URL into components (scheme, hostname, path, query) — same as extract_features.py

import pandas as pd                # for creating the single-row feature DataFrame that the classifiers expect
import numpy as np                 # for Shannon entropy calculation and numerical operations


logging.basicConfig(               # configure the logging system to show timestamps and log levels in the terminal
    level=logging.INFO,            # show INFO and above (INFO, WARNING, ERROR, CRITICAL) — same level as all other pipeline scripts
    format="%(asctime)s [%(levelname)s] %(message)s",  # include timestamp, log level, and message in every log line
    datefmt="%H:%M:%S",            # show only time (hours:minutes:seconds) in logs for readability
)
log = logging.getLogger(__name__)  # create a logger specifically for this module (src/predict.py)


# --------------------------------------------------------------
# File paths
# These must match the paths used by train_models.py when saving
# the model artifacts. If train_models.py saves to a different
# location, update these constants to match.
# --------------------------------------------------------------
OUTPUT_DIR  = "outputs"                                          # directory where train_models.py saves all artifacts
MODEL_LR    = os.path.join(OUTPUT_DIR, "model_lr.pkl")          # path to the saved Logistic Regression model
MODEL_RF    = os.path.join(OUTPUT_DIR, "model_rf.pkl")          # path to the saved Random Forest model
MODEL_DT    = os.path.join(OUTPUT_DIR, "model_dt.pkl")          # path to the saved Decision Tree model
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")            # path to the saved StandardScaler (fitted on training data by train_models.py)

# Human-readable display names for each model, used in the output table
MODEL_DISPLAY_NAMES = {
    "lr": "Logistic Regression",    # display name for the Logistic Regression model
    "rf": "Random Forest",           # display name for the Random Forest model
    "dt": "Decision Tree (depth=10)" # display name for the Decision Tree model, including the depth constraint for clarity
}


# --------------------------------------------------------------
# Feature engineering
# These functions are intentionally identical to those in
# extract_features.py. They must remain synchronised — if the
# feature set in extract_features.py changes, these functions
# must be updated to match, and train_models.py must be rerun
# to produce new .pkl files trained on the new feature schema.
# Loading old .pkl files with a new feature schema will cause
# a dimensionality mismatch error (fail-fast, not silent corruption).
# --------------------------------------------------------------
def _shannon_entropy(s: str) -> float:
    """
    Compute character-level Shannon entropy for a string.
    Formula: H = -sum(p(x) * log2(p(x))) for each unique character x.
    Higher entropy indicates more randomness in the character distribution,
    which is a common trait of algorithmically generated phishing URLs.
    Returns 0.0 for empty strings to avoid errors on empty URL components.
    Identical to the _shannon_entropy function in extract_features.py.
    """
    if not s:                                                      # if the string is empty, entropy is defined as 0 — avoids log(0) errors
        return 0.0                                                 # return 0.0 immediately for empty strings without performing any calculation
    freq = pd.Series(list(s)).value_counts(normalize=True)        # count how often each character appears and normalise to proportions (probabilities)
    return float(-(freq * np.log2(freq)).sum())                   # apply the Shannon entropy formula: H = -sum(p * log2(p)) and return as a Python float


def extract_features_single(url: str) -> pd.DataFrame:
    """
    Engineer all 24 URL-string features from a single URL string.
    Returns a single-row pandas DataFrame with 24 numeric columns,
    exactly matching the schema expected by the trained classifiers.

    This function replicates the feature engineering logic from
    extract_features.py applied to one URL instead of a full dataset.
    The output column order must match the order used during training,
    which is guaranteed because the column names are identical and
    scikit-learn models select features by column position at fit time.

    No network connections are made — all features are derived purely
    from the URL string using string operations, regex, and entropy.
    """
    url = str(url).strip()                                         # ensure the input is a string and remove any leading/trailing whitespace

    # ── Parse the URL into components using urllib.parse ─────────
    # If the URL has no scheme (e.g., "example.com/path"), prepend
    # "http://" so urlparse can correctly identify the hostname and path.
    # This mirrors the safe_parse logic in extract_features.py exactly.
    parsed   = urlparse(url if "://" in url else "http://" + url) # parse the URL; prepend http:// only if no scheme is present to ensure correct component extraction
    hostname = parsed.hostname or ""                               # extract the hostname component; use empty string if None (e.g., for malformed URLs) to prevent errors downstream
    path     = parsed.path     or ""                               # extract the path component; use empty string if None
    query    = parsed.query    or ""                               # extract the query string component (everything after '?'); use empty string if None

    # ── GROUP A: Length measurements (4 features) ─────────────────
    url_length      = len(url)                                     # total character length of the full URL string including scheme, hostname, path, and query
    hostname_length = len(hostname)                                # character length of the hostname component only (e.g., "www.example.com")
    path_length     = len(path)                                    # character length of the URL path component (e.g., "/login/verify")
    query_length    = len(query)                                   # character length of the query string after '?' (e.g., "id=123&session=abc")

    # ── GROUP B: Character count features (10 features) ──────────
    # Count specific characters in the full URL string.
    # These counts capture obfuscation patterns common in phishing URLs.
    num_dots        = url.count(".")                               # count dots — many dots indicate subdomain chains or encoded content common in phishing
    num_hyphens     = url.count("-")                               # count hyphens — excessive hyphens in hostnames are a phishing indicator
    num_underscores = url.count("_")                               # count underscores — less common in legitimate URLs, more common in obfuscated ones
    num_slashes     = url.count("/")                               # count forward slashes — deep paths with many slashes are common in malicious URLs
    num_qmarks      = url.count("?")                               # count question marks — multiple question marks indicate complex or malformed query strings
    num_equals      = url.count("=")                               # count equals signs — many equals signs indicate complex query parameters often used in phishing redirects
    num_ampersands  = url.count("&")                               # count ampersands — many ampersands indicate multiple query parameters, common in redirect-based phishing
    num_at          = url.count("@")                               # count @ symbols — the @ character in a URL causes browsers to treat everything before it as credentials, masking the real destination
    num_percent     = url.count("%")                               # count percent signs — percent-encoding (%XX) is used to obfuscate malicious characters in URLs
    num_digits      = sum(c.isdigit() for c in url)               # count all digit characters (0-9) in the full URL string — high digit counts can indicate IP addresses or generated URLs

    # ── GROUP C: Structural and boolean features (6 features) ────
    has_https    = int(parsed.scheme == "https")                   # 1 if the URL uses HTTPS scheme, 0 otherwise; note HTTPS alone does not guarantee legitimacy
    has_ip       = int(bool(                                       # 1 if the hostname is a raw IPv4 address (e.g., 192.168.1.1) instead of a domain name — a strong phishing indicator
        hostname and                                               # only check if hostname is not empty to avoid errors
        all(                                                       # check that every part of the hostname split by dots is a digit string between 0 and 255
            part.isdigit() and 0 <= int(part) <= 255              # each segment must be a valid octet (0-255)
            for part in hostname.split(".")                        # split hostname by dots to get individual segments
        ) and hostname.count(".") == 3                             # must have exactly 3 dots for a valid IPv4 address (e.g., a.b.c.d)
    ))
    has_port     = int(bool(parsed.port))                          # 1 if the URL specifies an explicit port number (e.g., :8080), 0 otherwise — non-standard ports are a phishing indicator
    has_www      = int(hostname.startswith("www."))                # 1 if the hostname begins with "www.", 0 otherwise — many phishing URLs omit www or use subdomains instead
    has_at_in_url = int("@" in url)                                # 1 if the @ symbol appears anywhere in the URL string — duplicates the character count but provides a direct binary flag for the classifier
    subdomain_depth = max(0, hostname.count(".") - 1) if hostname else 0  # number of subdomain levels above the registered domain; calculated as dots in hostname minus 1 (e.g., "a.b.example.com" has depth 2); 0 if hostname is empty

    # ── GROUP D: Entropy and ratio features (4 features) ─────────
    url_entropy      = _shannon_entropy(url)                       # Shannon entropy of the full URL character distribution — higher values indicate more random/generated URLs
    hostname_entropy = _shannon_entropy(hostname)                  # Shannon entropy of the hostname component only — high hostname entropy strongly indicates algorithmically generated domain names

    url_len_safe     = url_length if url_length > 0 else np.nan   # use NaN for division if URL length is 0 to avoid ZeroDivisionError; fillna(0) is applied at the end, matching extract_features.py behaviour

    digit_ratio      = num_digits / url_len_safe                   # proportion of URL characters that are digits; high digit ratio indicates IP-based or generated URLs
    special_char_ratio = (                                         # proportion of URL characters that are special characters — computed from exactly the same 8 character types as extract_features.py
        (num_dots + num_hyphens + num_underscores + num_at +       # sum of the 8 specific character counts that define special_char_ratio in this project (not all non-alphanumeric)
         num_percent + num_qmarks + num_equals + num_ampersands)   # these are the same 8 types used in extract_features.py: dots, hyphens, underscores, @, %, ?, =, &
        / url_len_safe                                             # divide by URL length to get the proportion; url_len_safe prevents division by zero
    )

    # ── Assemble all 24 features into a single-row DataFrame ─────
    # Column order matches extract_features.py output exactly.
    # scikit-learn models use positional feature order from training,
    # so this order must not be changed without retraining.
    features = {
        # Group A — Length features (4)
        "url_length":        url_length,       # total URL character count
        "hostname_length":   hostname_length,  # hostname character count
        "path_length":       path_length,      # path component character count
        "query_length":      query_length,     # query string character count
        # Group B — Character count features (10)
        "num_dots":          num_dots,         # count of '.' characters
        "num_hyphens":       num_hyphens,      # count of '-' characters
        "num_underscores":   num_underscores,  # count of '_' characters
        "num_slashes":       num_slashes,      # count of '/' characters
        "num_qmarks":        num_qmarks,       # count of '?' characters
        "num_equals":        num_equals,       # count of '=' characters
        "num_ampersands":    num_ampersands,   # count of '&' characters
        "num_at":            num_at,           # count of '@' characters
        "num_percent":       num_percent,      # count of '%' characters
        "num_digits":        num_digits,       # count of digit characters (0-9)
        # Group C — Structural/boolean features (6)
        "has_https":         has_https,        # 1 if scheme is https, else 0
        "has_ip":            has_ip,           # 1 if hostname is IPv4 address, else 0
        "has_port":          has_port,         # 1 if explicit port present, else 0
        "has_www":           has_www,          # 1 if hostname starts with www., else 0
        "has_at_in_url":     has_at_in_url,    # 1 if @ appears anywhere in URL, else 0
        "subdomain_depth":   subdomain_depth,  # number of subdomain levels above registered domain
        # Group D — Entropy and ratio features (4)
        "url_entropy":       url_entropy,      # Shannon entropy of full URL character distribution
        "hostname_entropy":  hostname_entropy, # Shannon entropy of hostname character distribution
        "digit_ratio":       digit_ratio,      # proportion of URL characters that are digits
        "special_char_ratio": special_char_ratio, # proportion of URL characters that are special characters (8 specific types)
    }

    df = pd.DataFrame([features])              # wrap the feature dictionary in a list to create a single-row DataFrame, which is the format scikit-learn models expect for prediction
    df = df.fillna(0)                          # replace any NaN values (from division by zero in ratio features for zero-length URLs) with 0, matching the fillna(0) behaviour in extract_features.py
    return df                                  # return the 24-column single-row DataFrame ready for classifier inference


# --------------------------------------------------------------
# Model loading
# --------------------------------------------------------------
def load_models() -> tuple:
    """
    Load all three trained classifier objects and the StandardScaler
    from the .pkl files saved by train_models.py.

    Exits with a clear error message if any file is missing,
    instructing the user to run train_models.py first.

    Returns: (lr_model, rf_model, dt_model, scaler)
    """
    required = {                               # dictionary mapping each required artifact to its file path for clear error reporting if any are missing
        "Logistic Regression": MODEL_LR,       # path to the Logistic Regression model artifact
        "Random Forest":       MODEL_RF,       # path to the Random Forest model artifact
        "Decision Tree":       MODEL_DT,       # path to the Decision Tree model artifact
        "StandardScaler":      SCALER_PATH,    # path to the StandardScaler artifact (must be the scaler fitted on training data only)
    }

    missing = [                                # build a list of any artifact files that do not exist on disk
        f"  {name}: {path}"                    # format each missing file as "  ModelName: path/to/file.pkl" for clear error output
        for name, path in required.items()     # iterate through all required artifacts and check whether each file exists
        if not os.path.exists(path)            # include in the missing list only if the file does not exist at the expected path
    ]

    if missing:                                # if any required files are missing, log a clear error and exit rather than producing a cryptic pickle error
        log.error(                             # log an error message listing all missing files
            "Required model files not found:\n%s\n"
            "Run train_models.py first to generate these files.",
            "\n".join(missing)                 # join all missing file descriptions with newlines for readable output
        )
        sys.exit(1)                            # exit with error code 1 — this is a fail-fast check, not a silent failure

    log.info("Loading model artifacts from %s/", OUTPUT_DIR)  # log that we are starting to load the model files from the outputs directory

    with open(MODEL_LR, "rb") as f:            # open the Logistic Regression .pkl file in binary read mode ("rb") for unpickling
        lr = pickle.load(f)                    # deserialise the Logistic Regression model object from the .pkl file

    with open(MODEL_RF, "rb") as f:            # open the Random Forest .pkl file in binary read mode for unpickling
        rf = pickle.load(f)                    # deserialise the Random Forest model object from the .pkl file

    with open(MODEL_DT, "rb") as f:            # open the Decision Tree .pkl file in binary read mode for unpickling
        dt = pickle.load(f)                    # deserialise the Decision Tree model object from the .pkl file

    with open(SCALER_PATH, "rb") as f:         # open the StandardScaler .pkl file in binary read mode for unpickling
        scaler = pickle.load(f)                # deserialise the StandardScaler object; this scaler was fitted on training data only and must be applied (transform only) to preserve the no-leakage guarantee

    log.info("All model artifacts loaded successfully.")       # log confirmation that all four artifacts were loaded without errors
    return lr, rf, dt, scaler                  # return all four loaded objects as a tuple for use in the classify function


# --------------------------------------------------------------
# Classification
# --------------------------------------------------------------
def classify(url: str, lr, rf, dt, scaler) -> None:
    """
    Run a single URL through all three classifiers and display
    a plain-language verdict for each model and a majority-vote
    summary. No output files are written — results are displayed
    to the terminal only.

    The StandardScaler is applied (transform only, never fit)
    to the feature vector before Logistic Regression inference,
    exactly as was done during training in train_models.py.
    Random Forest and Decision Tree receive unscaled features,
    also matching training behaviour.
    """
    log.info("Extracting 24 URL-string features...")           # log that feature extraction is starting so the user can see progress

    features_df = extract_features_single(url)                 # compute all 24 features from the input URL string using the same logic as extract_features.py

    log.info("Running classifiers...")                         # log that classification is about to begin across all three models

    # Apply the scaler to the feature vector for Logistic Regression only.
    # transform() is used — NOT fit_transform() — because the scaler
    # was already fitted on training data. Re-fitting here would
    # use the single test URL's statistics, which is data leakage.
    features_scaled = scaler.transform(features_df)            # transform the feature vector using the pre-fitted scaler; this standardises features to the same scale seen during training without re-fitting

    # ── Logistic Regression ───────────────────────────────────────
    lr_pred  = lr.predict(features_scaled)[0]                  # predict class label (0 or 1) for the scaled feature vector; [0] extracts the scalar value from the single-element array returned by predict()
    lr_proba = lr.predict_proba(features_scaled)[0]            # get class probabilities [P(legitimate), P(malicious)] for the scaled feature vector; [0] extracts the probabilities for the single sample

    # ── Random Forest ─────────────────────────────────────────────
    rf_pred  = rf.predict(features_df)[0]                      # predict class label using unscaled features — Random Forest does not require feature scaling as it uses decision thresholds not distance metrics
    rf_proba = rf.predict_proba(features_df)[0]                # get class probabilities for the unscaled feature vector

    # ── Decision Tree ─────────────────────────────────────────────
    dt_pred  = dt.predict(features_df)[0]                      # predict class label using unscaled features — Decision Tree also does not require scaling for the same reason as Random Forest
    dt_proba = dt.predict_proba(features_df)[0]                # get class probabilities for the unscaled feature vector

    # ── Majority vote ─────────────────────────────────────────────
    # Sum the three binary predictions (each is 0 or 1).
    # If the sum is 2 or 3, at least two of the three models predict
    # malicious, so the majority vote is Malicious.
    # If the sum is 0 or 1, the majority vote is Legitimate.
    votes       = lr_pred + rf_pred + dt_pred                  # sum the three binary predictions; result is 0, 1, 2, or 3 malicious votes
    majority    = "Malicious" if votes >= 2 else "Legitimate"  # majority verdict: Malicious if 2 or 3 models agree, Legitimate if 0 or 1 models predict malicious

    # ── Display results ───────────────────────────────────────────
    print("\n" + "=" * 60)                                      # print a separator line for visual clarity in the terminal output
    print("  PHISHING DETECTION — URL CLASSIFICATION RESULT")   # print the section header
    print("=" * 60)                                             # print another separator line
    print(f"  URL: {url}")                                      # display the URL that was classified so the user can confirm which URL the results refer to
    print("-" * 60)                                             # print a separator before the per-model results

    # Helper to format a single model result line
    def result_line(name, pred, proba):                         # inner function to format a single model's verdict line consistently for all three models
        verdict    = "Malicious" if pred == 1 else "Legitimate" # convert the binary prediction (1 or 0) to a human-readable verdict string
        confidence = proba[int(pred)] * 100                     # extract the confidence score for the predicted class as a percentage; proba[0] = P(Legitimate), proba[1] = P(Malicious)
        return f"  {name:<28s} {verdict:<12s}  ({confidence:.1f}% confidence)"  # format the line with the model name left-aligned in 28 characters, verdict left-aligned in 12 characters, and confidence to 1 decimal place

    print(result_line("Logistic Regression",     lr_pred, lr_proba))  # print the Logistic Regression result
    print(result_line("Random Forest",           rf_pred, rf_proba))  # print the Random Forest result
    print(result_line("Decision Tree (depth=10)", dt_pred, dt_proba)) # print the Decision Tree result
    print("-" * 60)                                             # print a separator before the majority vote summary
    print(f"  Majority Vote (2 of 3):      {majority}")         # print the majority-vote summary classification — the overall verdict from all three models combined
    print("=" * 60)                                             # print a final separator line

    # ── Additional note if models disagree ───────────────────────
    if votes == 1 or votes == 2:                                # if models disagree (1 or 2 malicious votes out of 3), print a caution note
        print(                                                  # print a caution message when classifiers disagree — this is important for User Class 3 (Corporate Security Team) who should escalate mixed results
            "\n  NOTE: Classifiers disagree on this URL. "
            "Treat the majority verdict with caution and\n"
            "  consider manual review before acting on this result."
        )

    print()                                                     # print a blank line after the results for clean terminal output


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------
def parse_args() -> argparse.Namespace:                        # function to parse command-line arguments; only one argument is required — the URL to classify
    """Parse the --url command-line argument."""
    p = argparse.ArgumentParser(                               # create an ArgumentParser to handle command-line arguments and automatically generate --help output
        description=(                                          # description shown when user runs: python src/predict.py --help
            "Classify a single URL as Malicious or Legitimate "
            "using trained model artifacts from train_models.py. "
            "No network connections are made during classification."
        )
    )
    p.add_argument(                                            # define the --url argument that the user must supply
        "--url",                                               # flag name: user runs the script as: python src/predict.py --url https://example.com
        required=True,                                         # this argument is mandatory — the script cannot run without a URL to classify
        help=(                                                 # help text shown in --help output explaining what to provide
            "The URL to classify. "
            "Example: --url https://example.com"
        ),
    )
    return p.parse_args()                                      # parse the arguments from the command line and return them as a Namespace object with a .url attribute


# --------------------------------------------------------------
# Entry point
# --------------------------------------------------------------
def main() -> None:                                            # main function that orchestrates argument parsing, model loading, and classification
    """
    Entry point for predict.py. Parses the --url argument,
    loads model artifacts, extracts features, and displays results.
    """
    args = parse_args()                                        # parse command-line arguments to retrieve the URL the user wants to classify

    log.info("=" * 60)                                         # log a separator line for visual clarity at the start of execution
    log.info("predict.py  --  URL inference only")             # log the script name and purpose so the user knows which script is running
    log.info("=" * 60)                                         # log another separator line

    lr, rf, dt, scaler = load_models()                         # load all three trained classifier objects and the StandardScaler from the .pkl files in outputs/; exits with error if any file is missing

    classify(args.url, lr, rf, dt, scaler)                     # run the URL through all three classifiers and display the plain-language verdict; no output files are written


if __name__ == "__main__":                                     # standard Python idiom: only execute main() if this script is run directly, not if it is imported as a module by another script
    try:
        main()                                                 # call the main function to execute the full inference pipeline: parse args → load models → extract features → classify → display results
    except Exception:
        log.error("Unhandled error:\n%s", traceback.format_exc())  # catch any unexpected exception, log it with a full traceback for debugging, and exit cleanly
        sys.exit(1)                                            # exit with error code 1 to indicate that an unhandled error occurred during execution