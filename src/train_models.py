# -*- coding: utf-8 -*-
# --------------------------------------------------------------
# src/train_models.py
# --------------------------------------------------------------
# Purpose:
#   Train and evaluate 3 lightweight ML models on phishing URL
#   datasets and display a visual dashboard for each dataset.
#
#   Models:
#       1. Logistic Regression
#       2. Random Forest
#       3. Decision Tree
#
#   For each dataset it:
#       - Loads the feature CSV produced by extract_features.py
#       - Splits data into stratified train/test sets
#       - Trains and evaluates each model
#       - Displays a dashboard with:
#           * Metrics table (accuracy, precision, recall, F1)
#           * Bar chart comparing all 3 models
#           * Confusion matrix heatmaps
#           * Feature importance chart (Random Forest)
#       - Saves dashboard as PNG to outputs/
#       - Saves all results to outputs/model_results.csv
#       - Saves full terminal output to outputs/training_log_<timestamp>.txt
#
# Input files (produced by extract_features.py):
#   data/features_phishtank_tranco_dataset.csv
#   data/features_urlhaus_tranco_dataset.csv
#
# Usage:
#   python src/train_models.py
#   python src/train_models.py --testsize 0.2
# --------------------------------------------------------------

import os # For file handling and path operations
import sys # For stdout redirection and system exit
import logging # For structured logging of results and errors
import argparse # For command-line argument parsing
import traceback # For detailed error tracebacks in logs

import pandas as pd # For data manipulation and analysis
import numpy as np # For numerical operations and array handling
import matplotlib.pyplot as plt # For plotting the results dashboard
import matplotlib.gridspec as gridspec  # For advanced subplot layouts in the dashboard
import seaborn as sns # For enhanced data visualization (confusion matrices, feature importance)
import pickle # For saving trained model artifacts to disk

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score # For splitting data and performing cross-validation
from sklearn.preprocessing    import StandardScaler # For feature scaling (needed for Logistic Regression)
from sklearn.linear_model     import LogisticRegression # For training the Logistic Regression model
from sklearn.ensemble         import RandomForestClassifier # For training the Random Forest model and extracting feature importances
from sklearn.tree             import DecisionTreeClassifier # For training the Decision Tree model
from sklearn.metrics          import ( # For evaluating model performance with various metrics and reports
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report,
)

class _Tee: 
    """
    Mirrors all writes to stdout into a file at the same time.
    This ensures both print() output and log output are captured.
    """
    def __init__(self, file_path: str): # Initialize the tee by opening the target file and keeping a reference to the original stdout
        self._terminal = sys.stdout # Keep original stdout to still print to terminal
        self._file     = open(file_path, "w", encoding="utf-8") # Open the file for writing logs

    def write(self, message: str): # Write the message to both the terminal and the file
        self._terminal.write(message) # Write to terminal
        self._file.write(message) # Write to file

    def flush(self): # Flush both the terminal and file buffers to ensure all output is written out immediately
        self._terminal.flush() # Flush terminal buffer
        self._file.flush() # Flush file buffer

    def close(self): # Close the file when done to free resources
        self._file.close() # Close the file to ensure all data is saved and resources are released


def setup_logging(output_dir: str) -> str: # Set up logging to capture all output to a timestamped file in the specified output directory
    """
    Configure logging and redirect stdout so that EVERYTHING printed
    during training — metrics, classification reports, feature
    importances, CV scores, and the results summary — is saved to a
    timestamped text file in outputs/ as well as shown in the terminal.

    Saved to: outputs/training_results_YYYYMMDD_HHMMSS.txt
    """
    import datetime # For generating a timestamp for the log filename
    os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists, creating it if necessary

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Generate a timestamp string for the log filename
    log_path  = os.path.join(output_dir, f"training_results_{timestamp}.txt") # Construct the full path for the log file using the output directory and timestamped filename

    # Redirect stdout — captures all print() calls
    sys.stdout = _Tee(log_path) # Redirect standard output to the _Tee class, which will write to both the terminal and the log file

    # Configure logging to also write to the same stdout (now tee'd)
    fmt    = "%(asctime)s [%(levelname)s] %(message)s" # Define the logging format to include timestamp, log level, and message
    logger = logging.getLogger() # Get the root logger to configure global logging settings
    logger.setLevel(logging.INFO) # Set the logging level to INFO to capture all informational messages and above (warnings, errors)
    handler = logging.StreamHandler(sys.stdout) # Create a logging handler that writes to the same stdout (which is now tee'd to the log file)
    handler.setFormatter(logging.Formatter(fmt, "%H:%M:%S")) # Set the formatter for the logging handler to use the defined format and only show time (not date) in the timestamp
    logger.addHandler(handler) # Add the configured handler to the logger so that all log messages will be processed by this handler and written to the tee'd stdout

    logging.getLogger(__name__).info( # Log an informational message indicating where the results will be saved, using the absolute path to the log file for clarity
        "Results will be saved -> %s", os.path.abspath(log_path) # Log the absolute path to the log file where all results will be saved, ensuring users know exactly where to find the output
    )
    return log_path # Return the path to the log file for reference if needed elsewhere in the code


log = logging.getLogger(__name__) # Create a logger for this module to use for logging messages throughout the code, allowing for consistent and configurable logging behavior

# --------------------------------------------------------------
# Feature files — produced by extract_features.py
# Update these paths if your filenames differ
# --------------------------------------------------------------
FEATURE_FILES = [ # List of feature CSV files to load and process, produced by the extract_features.py script. Update these paths if your filenames or locations differ.
    os.path.join("data", "features_phishtank_tranco_dataset.csv"),
    os.path.join("data", "features_urlhaus_tranco_dataset.csv"),
]

OUTPUT_DIR       = "outputs" # Directory where all outputs (dashboards, results CSV, logs) will be saved. The code will create this directory if it doesn't exist.
LABEL_CANDIDATES = ["label", "class", "target", "type", "result", "status"] # Possible column names in the dataset that may contain the label (malicious vs legitimate). The code will search for these column names (case-insensitive) to identify which column contains the labels.
LABEL_MAP        = { # Mapping of various label values to a standardized binary format (1 for malicious, 0 for legitimate). This allows the code to handle different datasets that may use different conventions for labeling.
    "phishing": 1, "phish": 1, "bad": 1, "malicious": 1, "spam": 1, "1": 1,
    "legitimate": 0, "benign": 0, "good": 0, "safe": 0, "0": 0,
}

METRIC_COLOURS = ["#2E75B6", "#2EA043", "#E67E22", "#8E44AD"] # Colors for the metrics in the dashboard bar chart (accuracy, precision, recall, F1). These colors will be used consistently across all dashboards for visual clarity and branding.


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame: # Load a dataset from a CSV file, trying different encodings if necessary to handle potential Unicode issues in the data
    try: # First attempt to load the dataset using the default encoding, which works for most CSV files
        return pd.read_csv(path, low_memory=False) # Load the CSV file into a pandas DataFrame, using low_memory=False to optimize memory usage for large files
    except UnicodeDecodeError: # If a UnicodeDecodeError occurs, it likely means the file contains characters that can't be decoded with the default encoding. In this case, we try again with a different encoding (latin-1) which can handle a wider range of characters without throwing an error.
        return pd.read_csv( # Load the CSV file again, this time specifying encoding="latin-1" to handle potential Unicode issues. We also set on_bad_lines="skip" to skip any lines that still cause issues, and low_memory=False for better performance.
            path, encoding="latin-1", # Specify the path to the CSV file and the encoding to use for reading it
            on_bad_lines="skip", low_memory=False # Handle bad lines by skipping them and optimize memory usage for large files
        )


def find_label_column(df: pd.DataFrame) -> str | None: # Identify which column in the DataFrame contains the label (malicious vs legitimate) by checking for common candidate column names in a case-insensitive manner
    lower_map = {c.lower(): c for c in df.columns} # Create a mapping of lowercase column names to their original case versions, allowing for case-insensitive searching of column names
    for key in LABEL_CANDIDATES: # Iterate through the list of candidate column names that may contain the label, checking if any of them (in lowercase) exist in the DataFrame's columns
        if key in lower_map: # If a candidate column name (in lowercase) is found in the mapping, it means we have identified the label column. We return the original case version of the column name from the mapping to ensure we can access it correctly in the DataFrame.
            return lower_map[key] # Return the original case version of the column name that contains the labels, allowing the rest of the code to access this column correctly when processing the dataset
    return None # If no label column is found after checking all candidates, return None to indicate that the dataset does not contain a recognizable label column


def normalize_labels(series: pd.Series) -> pd.Series: # Normalize the label values in the series to a binary format (1 for malicious, 0 for legitimate) using the LABEL_MAP. If the series is numeric, we convert it to numeric and coerce errors to NaN. If it's not numeric, we convert it to string, strip whitespace, convert to lowercase, and then map using LABEL_MAP.
    if pd.api.types.is_numeric_dtype(series): # Check if the series is already of a numeric data type (e.g., int, float). If it is, we will attempt to convert it to numeric values, coercing any errors to NaN. This allows us to handle cases where the labels might be numeric but contain some invalid entries that can't be converted to numbers.
        return pd.to_numeric(series, errors="coerce") # If the series is already numeric, we attempt to convert it to numeric values, coercing any errors to NaN. This allows us to handle cases where the labels might be numeric but contain some invalid entries.
    return series.astype(str).str.strip().str.lower().map(LABEL_MAP) # If the series is not numeric, we convert it to string, strip any leading/trailing whitespace, convert it to lowercase, and then map the values using the LABEL_MAP to convert various label formats into a standardized binary format (1 for malicious, 0 for legitimate). This allows us to handle a wide variety of label formats across different datasets and ensure consistency in how we interpret the labels for training and evaluation.


# --------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------
def show_dashboard( # Build and display a professional results dashboard for one dataset, showing metrics, comparisons, confusion matrices, and feature importances. Saves a PNG copy to the outputs/ directory automatically.
    dataset_name: str, # The name of the dataset being evaluated, used for titling the dashboard and naming the output file
    model_results: list, # A list of dictionaries containing the results for each model (accuracy, precision, recall, F1, confusion matrix values) that will be displayed in the dashboard
    importances: pd.Series, # A pandas Series containing the feature importances from the Random Forest model, which will be displayed in the dashboard
    output_dir: str, # The directory where the dashboard PNG will be saved. The code will create this directory if it doesn't exist, and the dashboard will be saved with a filename that includes the dataset name for clarity.
) -> None: # This function does not return anything; it generates and displays the dashboard, and saves it to disk.
    """
    Build and display a professional results dashboard for one dataset.
    Saves a PNG copy to outputs/ automatically.
    """
    sns.set_theme(style="whitegrid", font_scale=0.9) # Set the Seaborn theme for the plots in the dashboard to "whitegrid" for a clean look, and set the font scale to 0.9 for slightly smaller text that fits well in the dashboard layout. This ensures that all visual elements in the dashboard have a consistent and professional appearance.

    fig = plt.figure(figsize=(20, 14)) # Create a new figure for the dashboard with a specified size (20 inches wide by 14 inches tall) to provide ample space for all the subplots and visual elements that will be included in the dashboard, such as the metrics table, bar charts, confusion matrices, and feature importance chart.
    fig.patch.set_facecolor("#0D1117") # Set the background color of the entire figure to a dark shade (#0D1117) to create a visually appealing contrast with the plots and make the colors of the bars, tables, and heatmaps stand out more effectively in the dashboard.

    clean_name = ( # Create a clean, human-readable title for the dashboard by taking the dataset name, removing any prefixes like "features_", replacing underscores with spaces, and converting it to title case. This makes the dashboard title more understandable and visually appealing when displayed at the top of the figure.
        dataset_name # Take the dataset name (e.g., "features_phishtank_tranco_dataset") and clean it up for display
        .replace("features_", "") # Remove any "features_" prefix from the dataset name to make it cleaner for display in the dashboard title
        .replace("_", " ") # Replace underscores with spaces to improve readability in the dashboard title (e.g., "phishtank tranco dataset" instead of "phishtank_tranco_dataset")
        .title() # Convert the cleaned dataset name to title case (e.g., "Phishtank Tranco Dataset") for a more professional and visually appealing dashboard title
    )
    fig.suptitle( # Set the overall title for the dashboard figure, using the cleaned dataset name to indicate which dataset's results are being displayed. The title is styled with a larger font size, bold weight, and white color to make it stand out against the dark background of the dashboard.
        f"Phishing Detection - {clean_name}", # Set the title of the dashboard to include "Phishing Detection" and the cleaned dataset name for clarity on what the dashboard is showing
        fontsize=16, fontweight="bold", # Style the title with a larger font size and bold weight to make it prominent in the dashboard
        color="white", y=0.98, # Set the title color to white for contrast against the dark background, and position it slightly above the top of the figure (y=0.98) to give it some space from the top edge of the dashboard
    )

    gs = gridspec.GridSpec( # Create a GridSpec layout for the subplots in the dashboard, defining a grid with
        3, 4, figure=fig, # 3 rows and 4 columns to organize the different visual elements (metrics table, bar chart, confusion matrices, feature importance) in a structured way. The layout is designed to have the metrics table span the full width of the top row, the bar chart and feature importance side by side in the second row, and the confusion matrices arranged in the third row.
        hspace=0.55, wspace=0.4, # Set the horizontal and vertical spacing between subplots to ensure they are visually separated and not too cramped, allowing for clear readability of all elements in the dashboard.
        left=0.06, right=0.97, # Set the left and right margins of the entire grid to provide some padding around the edges of the dashboard, ensuring that the plots and tables do not touch the edges of the figure and have a clean, professional appearance.
        top=0.92, bottom=0.06, # Set the top and bottom margins of the grid to provide space for the overall title at the top and some padding at the bottom, ensuring that all elements in the dashboard are well-positioned and visually balanced within the figure.
    )

    model_names   = [r["model"] for r in model_results] # Extract the model names from the results to use in the dashboard for labeling the bar charts and confusion matrices, ensuring that each visual element is clearly associated with the correct model for easy interpretation of the results.
    metrics       = ["accuracy", "precision", "recall", "f1_score"] # Define the list of metrics that will be displayed in the dashboard, corresponding to the keys in the model_results dictionaries. These metrics will be used to create the grouped bar chart comparing the performance of the different models across these key evaluation metrics.
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"] # Define the human-readable labels for the metrics that will be displayed in the dashboard, corresponding to the keys in the model_results. These labels will be used in the bar chart legend and axis labels to make it clear to viewers what each metric represents in a more understandable format (e.g., "F1-Score" instead of "f1_score").

    # ── Row 0: Metrics table (full width) ───────────────────────
    ax_table = fig.add_subplot(gs[0, :]) # Add a subplot for the metrics table that spans the entire first row of the grid (all 4 columns) to provide ample space for displaying the performance metrics of all models in a clear and organized manner at the top of the dashboard.
    ax_table.set_facecolor("#161B22") # Set the background color of the metrics table subplot to a slightly lighter dark shade (#161B22) to create a distinct area for the table that contrasts with the overall figure background, making the table visually stand out in the dashboard.
    ax_table.axis("off") # Turn off the axis for the table subplot since we will be displaying a table instead of a traditional plot, ensuring that no unnecessary axes or ticks are shown around the table for a cleaner and more professional appearance.

    table_data = [ # Prepare the data for the metrics table by creating a list of lists, where each inner list represents a row in the table. Each row contains the model name, accuracy, precision, recall, F1-score, and the total number of test rows (calculated from the confusion matrix values) for that model. The metric values are formatted to 4 decimal places for consistency and readability in the table.
        [
            r["model"], # The name of the model (e.g., "Logistic Regression", "Random Forest") to be displayed in the first column of the table for easy identification of which metrics correspond to which model.
            f"{r['accuracy']:.4f}", # The accuracy metric for the model, formatted to 4 decimal places, to be displayed in the second column of the table for a clear and precise representation of the model's performance in terms of overall correctness.
            f"{r['precision']:.4f}", # The precision metric for the model, formatted to 4 decimal places, to be displayed in the third column of the table for a clear and precise representation of the model's performance in terms of how many of the predicted positives were actually correct.
            f"{r['recall']:.4f}", # The recall metric for the model, formatted to 4 decimal places, to be displayed in the fourth column of the table for a clear and precise representation of the model's performance in terms of how many of the actual positives were correctly identified.
            f"{r['f1_score']:.4f}", # The F1-score metric for the model, formatted to 4 decimal places, to be displayed in the fifth column of the table for a clear and precise representation of the model's performance in terms of the harmonic mean of precision and recall.
            f"{r['tp']+r['tn']+r['fp']+r['fn']} rows", # The total number of test rows for the model, calculated by summing the true positives, true negatives, false positives, and false negatives from the confusion matrix. This value is displayed in the last column of the table to provide context on the size of the test set used for evaluating the model's performance.
        ]
        for r in model_results # Iterate through the list of model results to create a row in the table for each model, extracting the relevant metrics and confusion matrix values to populate the table data that will be displayed in the dashboard.
    ]
    col_labels = ["Model", "Accuracy", "Precision", # Define the column labels for the metrics table to be displayed at the top of the dashboard, providing clear and concise headers for each column that correspond to the data being shown (model name, accuracy, precision, recall, F1-score, and test rows).
                  "Recall", "F1-Score", "Test Rows"] # The column labels are chosen to be human-readable and informative, making it easy for viewers of the dashboard to understand what each column represents at a glance.

    tbl = ax_table.table( # Create a table in the metrics table subplot using the prepared data and column labels. The cell text is centered for better readability, and the table is positioned in the center of the subplot to create a balanced and visually appealing layout for the performance metrics of all models.
        cellText=table_data, # The data to be displayed in the table, organized as a list of lists where each inner list represents a row of the table containing the model name, accuracy, precision, recall, F1-score, and test rows for each model.
        colLabels=col_labels, # The column labels for the table, providing clear headers for each column that correspond to the data being shown (model name, accuracy, precision, recall, F1-score, and test rows).
        cellLoc="center", # Set the alignment of the text in the table cells to be centered, ensuring that all values in the table are neatly aligned and easy to read, contributing to the overall professional appearance of the dashboard.
        loc="center", # Position the table in the center of the subplot to create a balanced and visually appealing layout for the performance metrics of all models, making it the focal point of the top section of the dashboard.
    )
    tbl.auto_set_font_size(False) # Disable automatic font size adjustment for the table to allow for manual control over the font size, ensuring that the text in the table is large enough to be easily readable while still fitting well within the layout of the dashboard.
    tbl.set_fontsize(10) # Set the font size of the table text to 10 points, providing a good balance between readability and fitting the table within the available space in the dashboard, especially when there are multiple models and metrics being displayed.
    tbl.scale(1, 2.2) # Scale the table to adjust the row height, making it taller (2.2 times the default height) to improve readability and create more visual separation between rows, while keeping the column width unchanged (1 times the default width) to maintain a compact layout for the model names and metrics in the dashboard.

    for j in range(len(col_labels)): # Iterate through the column labels to style the header row of the table, setting a distinct background color and text properties to differentiate it from the data rows and make it visually stand out as the header of the metrics table in the dashboard.
        tbl[(0, j)].set_facecolor("#2E75B6") # Set the background color of the header row (first row, index 0) to a blue shade (#2E75B6) to visually differentiate it from the data rows and make it stand out as the header of the metrics table in the dashboard.
        tbl[(0, j)].set_text_props(color="white", fontweight="bold") # Set the text properties of the header row to have white color and bold weight, enhancing the visibility and emphasis of the column labels in the metrics table, making it clear to viewers what each column represents in the dashboard.

    row_colours = ["#1C2128", "#21262D"] # Define alternating row colors for the data rows in the table to improve readability and create a visually appealing striped effect, making it easier for viewers to distinguish between different rows of metrics for each model in the dashboard.
    for i in range(len(table_data)): # Iterate through the data rows of the table to apply alternating background colors and set text properties, enhancing the readability of the metrics for each model by creating a clear visual separation between rows in the dashboard.
        for j in range(len(col_labels)): # Iterate through the columns for each data row to apply the background color and text properties, ensuring that all cells in the data rows are styled consistently according to the defined alternating row colors for better readability in the dashboard.
            tbl[(i+1, j)].set_facecolor(row_colours[i % 2]) # Set the background color of the data rows (starting from index 1 since index 0 is the header) to alternate between two defined colors based on the row index, creating a striped effect that improves readability and helps viewers easily distinguish between different rows of metrics for each model in the dashboard.
            tbl[(i+1, j)].set_text_props(color="white") # Set the text color of the data rows to white to ensure good contrast against the dark background colors, enhancing the readability of the metrics for each model in the dashboard and maintaining a consistent visual style throughout the table.

    ax_table.set_title( # Set the title for the metrics table subplot, providing a clear and concise description of what the table represents (a summary of model performance metrics) to help viewers understand the context of the data being displayed in this section of the dashboard.
        "Model Performance Summary", # The title of the metrics table, indicating that this section of the dashboard provides a summary of the performance metrics for each model, helping viewers understand the context of the data being displayed in this part of the dashboard.
        color="white", fontsize=11, fontweight="bold", pad=8 # Style the title with white color for contrast against the dark background, a font size of 11 for visibility, bold weight for emphasis, and a padding of 8 to create some space between the title and the table for better visual separation in the dashboard.
    )

    # ── Row 1 Col 0-1: Grouped bar chart ────────────────────────
    ax_bar = fig.add_subplot(gs[1, :2]) # Add a subplot for the grouped bar chart that occupies the first two columns of the second row in the grid, providing enough space to display a clear comparison of the performance metrics (accuracy, precision, recall, F1) across all three models in a visually appealing way in the dashboard.
    ax_bar.set_facecolor("#161B22") # Set the background color of the bar chart subplot to a slightly lighter dark shade (#161B22) to create a distinct area for the bar chart that contrasts with the overall figure background, making the bars and text in the chart visually stand out in the dashboard.

    x      = np.arange(len(model_names)) # Create an array of x positions for the bars in the grouped bar chart, where each position corresponds to a model. This will be used to position the bars for each metric side by side for each model in the dashboard.
    width  = 0.18 # Set the width of each bar in the grouped bar chart to 0.18, allowing for multiple bars (one for each metric) to be displayed side by side for each model without overlapping, while still fitting well within the allocated space in the dashboard.
    offset = -(len(metrics) - 1) / 2 * width # Calculate the offset for the bars to ensure they are centered around the x positions for each model. This offset is based on the number of metrics and the width of the bars, allowing for an even distribution of bars around the central position for each model in the dashboard.

    for idx, (metric, label, colour) in enumerate( # Iterate through the metrics, their corresponding labels, and colors to create a bar for each metric in the grouped bar chart. The index (idx) is used to calculate the position of each bar for the different metrics, ensuring they are displayed side by side for each model in the dashboard.
        zip(metrics, metric_labels, METRIC_COLOURS) # Zip together the list of metric keys, their human-readable labels, and the corresponding colors to iterate through them in parallel when creating the bars for the grouped bar chart, ensuring that each metric is represented with the correct label and color in the dashboard.
    ):
        vals = [r[metric] for r in model_results] # Extract the values for the current metric from the model results to be used as the heights of the bars in the grouped bar chart, allowing for a visual comparison of this metric across all models in the dashboard.
        bars = ax_bar.bar( # Create a bar for the current metric in the grouped bar chart, positioning it based on the x positions for each model and the calculated offset to ensure the bars for different metrics are displayed side by side for each model. The bars are colored according to the specified color for this metric and have an alpha value of 0.88 for better visual appeal in the dashboard.
            x + offset + idx * width, vals, # Position the bars for this metric based on the x positions for each model, the calculated offset to center the bars, and the index to space them side by side. The height of each bar corresponds to the value of this metric for each model, allowing for a visual comparison of this metric across all models in the dashboard.
            width, label=label, color=colour, alpha=0.88 # Set the width of the bars, the label for the legend, the color for this metric, and the alpha for visual appeal in the dashboard
        )
        for bar, val in zip(bars, vals): # Iterate through the bars that were just created for this metric and their corresponding values to add text labels on top of each bar, showing the exact value of the metric for each model in the dashboard for better readability and interpretation of the results.
            ax_bar.text( # Add a text label on top of each bar to show the exact value of the metric for each model, formatted to 3 decimal places for clarity. The text is positioned at the center of the bar's width and slightly above the top of the bar for better visibility in the dashboard.
                bar.get_x() + bar.get_width() / 2, # Position the text at the center of the bar's width by taking the x position of the bar and adding half of the bar's width
                bar.get_height() + 0.01, # Position the text slightly above the top of the bar by taking the height of the bar and adding a small offset (0.01) to ensure it does not overlap with the bar itself, making it easier to read in the dashboard.
                f"{val:.3f}", # Format the value of the metric to 3 decimal places for clarity in the text label
                ha="center", va="bottom", # Set the horizontal alignment of the text to center and vertical alignment to bottom so that the text is centered above the bar and positioned just above it for better readability in the dashboard.
                fontsize=7, color="white", # Set the font size of the text label to 7 for better fit and readability above the bars, and set the color to white for good contrast against the dark background of the bar chart in the dashboard.
            )

    ax_bar.set_xticks(x) # Set the x-ticks of the bar chart to correspond to the positions of the models, ensuring that each group of bars is labeled with the correct model name in the dashboard for easy identification of which bars correspond to which models.
    ax_bar.set_xticklabels( # Set the x-tick labels to the model names, replacing spaces with newlines for better readability in the dashboard. This allows the model names to be displayed on multiple lines if they are long, preventing overlap and improving the overall appearance of the bar chart in the dashboard.
        [m.replace(" ", "\n") for m in model_names], # Replace spaces in the model names with newlines to allow for multi-line labels on the x-axis, improving readability and preventing overlap of long model names in the bar chart of the dashboard.
        color="white", fontsize=9 # Set the color of the x-tick labels to white for contrast against the dark background, and set the font size to 9 for better readability in the dashboard.
    )
    ax_bar.set_ylim(0, 1.15) # Set the y-axis limits of the bar chart to range from 0 to 1.15, providing some extra space above the maximum value of 1 for better visual appeal and to ensure that the text labels above the bars do not get cut off in the dashboard.
    ax_bar.set_ylabel("Score", color="white") # Set the y-axis label to "Score" to indicate that the values on the y-axis represent the performance scores (accuracy, precision, recall, F1) of the models. The label is colored white for contrast against the dark background of the bar chart in the dashboard.
    ax_bar.set_title( # Set the title for the bar chart subplot, providing a clear description of what the chart represents (a comparison of performance metrics across models) to help viewers understand the context of the data being displayed in this section of the dashboard.
        "Metrics Comparison", color="white", fontweight="bold"
    )
    ax_bar.tick_params(colors="white") # Set the color of the tick labels and ticks to white for contrast against the dark background, enhancing the readability of the axis labels and ticks in the bar chart of the dashboard.
    ax_bar.spines[:].set_color("#444") # Set the color of the spines (borders) of the bar chart to a dark gray (#444) to create a subtle border around the plot area that contrasts with the background without being too distracting, contributing to the overall professional appearance of the dashboard.
    ax_bar.legend( # Add a legend to the bar chart to indicate which color corresponds to which metric, providing clarity for viewers to understand what each bar represents in terms of the performance metrics being compared across the models in the dashboard. The legend is styled with a dark background and white text for consistency with the overall dashboard theme.
        loc="upper right", fontsize=8,
        facecolor="#21262D", labelcolor="white"
    )

    # ── Row 1 Col 2-3: Feature importance ───────────────────────
    ax_imp = fig.add_subplot(gs[1, 2:]) # Add a subplot for the feature importance chart that occupies the last two columns of the second row in the grid, providing enough space to display the top 10 feature importances from the Random Forest model in a clear and visually appealing horizontal bar chart in the dashboard.
    ax_imp.set_facecolor("#161B22") # Set the background color of the feature importance subplot to a slightly lighter dark shade (#161B22) to create a distinct area for the feature importance chart that contrasts with the overall figure background, making the bars and text in the chart visually stand out in the dashboard.
    
    top_n = importances.head(10) # Get the top 10 feature importances from the Random Forest model to be displayed in the horizontal bar chart, allowing viewers to see which features were most influential in the model's predictions in the dashboard.
    bars  = ax_imp.barh( # Create a horizontal bar chart for the top 10 feature importances, with the feature names on the y-axis and their corresponding importance scores on the x-axis. The bars are colored with a blue shade and have an alpha value of 0.85 for better visual appeal in the dashboard.
        top_n.index[::-1], top_n.values[::-1],
        color="#2E75B6", alpha=0.85
    )
    for bar, val in zip(bars, top_n.values[::-1]): # Iterate through the bars in the feature importance chart and their corresponding values to add text labels on top of each bar, showing the exact importance score for each feature in the dashboard for better readability and interpretation of the results.
        ax_imp.text( # Add a text label on top of each bar to show the exact importance score for each feature, formatted to 4 decimal places for clarity. The text is positioned at the end of the bar for better visibility in the dashboard.
            bar.get_width() + 0.001, # Position the text slightly to the right of the end of the bar by taking the width of the bar and adding a small offset (0.001) to ensure it does not overlap with the bar itself, making it easier to read in the dashboard.
            bar.get_y() + bar.get_height() / 2, # Position the text vertically at the center of the bar by taking the y position of the bar and adding half of the bar's height, ensuring that the text is aligned with the corresponding feature in the horizontal bar chart for better readability in the dashboard.
            f"{val:.4f}", va="center", # Set the vertical alignment of the text to center so that it is aligned with the corresponding feature in the horizontal bar chart, and format the value of the importance score to 4 decimal places for clarity in the text label.
            fontsize=8, color="white" # Set the font size of the text label to 8 for better fit and readability next to the bars, and set the color to white for good contrast against the dark background of the feature importance chart in the dashboard.
        )

    ax_imp.set_xlabel("Importance Score", color="white") # Set the x-axis label to "Importance Score" to indicate that the values on the x-axis represent the importance scores of the features as determined by the Random Forest model. The label is colored white for contrast against the dark background of the chart in the dashboard.
    ax_imp.set_title( # Set the title for the feature importance subplot, providing a clear description of what the chart represents (the top 10 feature importances from the Random Forest model) to help viewers understand the context of the data being displayed in this section of the dashboard.
        "Top 10 Feature Importances\n(Random Forest)",
        color="white", fontweight="bold"
    )
    ax_imp.tick_params(colors="white") # Set the color of the tick labels and ticks to white for contrast against the dark background, enhancing the readability of the axis labels and ticks in the feature importance chart of the dashboard.
    ax_imp.spines[:].set_color("#444") # Set the color of the spines (borders) of the feature importance chart to a dark gray (#444) to create a subtle border around the plot area that contrasts with the background without being too distracting, contributing to the overall professional appearance of the dashboard.
    ax_imp.set_facecolor("#161B22") # Set the background color of the feature importance subplot to a slightly lighter dark shade (#161B22) to create a distinct area for the feature importance chart that contrasts with the overall figure background, making the bars and text in the chart visually stand out in the dashboard.

    # ── Row 2: Confusion matrices ────────────────────────────────
    for idx, result in enumerate(model_results): # Iterate through the model results to create a confusion matrix heatmap for each model in the third row of the grid. Each confusion matrix will be displayed in its own subplot, allowing for a clear visual comparison of the true positives, true negatives, false positives, and false negatives for each model in the dashboard.
        ax_cm = fig.add_subplot(gs[2, idx]) # Add a subplot for the confusion matrix of the current model in the third row of the grid, with the column index corresponding to the model's position in the model_results list. This allows each model's confusion matrix to be displayed side by side for easy comparison in the dashboard.
        ax_cm.set_facecolor("#161B22") # Set the background color of the confusion matrix subplot to a slightly lighter dark shade (#161B22) to create a distinct area for the confusion matrix that contrasts with the overall figure background, making the heatmap and text in the confusion matrix visually stand out in the dashboard.

        cm = np.array([ # Create a 2x2 array for the confusion matrix using the true negatives, false positives, false negatives, and true positives from the model results. This array will be used to create a heatmap that visually represents the confusion matrix for each model in the dashboard.
            [result["tn"], result["fp"]], # The first row of the confusion matrix contains the true negatives (tn) and false positives (fp), representing the counts of legitimate samples correctly identified as legitimate and legitimate samples incorrectly identified as malicious, respectively.
            [result["fn"], result["tp"]], # The second row of the confusion matrix contains the false negatives (fn) and true positives (tp), representing the counts of malicious samples incorrectly identified as legitimate and malicious samples correctly identified as malicious, respectively.
        ])

        sns.heatmap( # Create a heatmap for the confusion matrix using Seaborn, with annotations to show the counts in each cell. The heatmap is colored using the "Blues" colormap, and the x and y tick labels are set to indicate "Legitimate" and "Malicious" for better readability in the dashboard.
            cm, annot=True, fmt="d", # Annotate the heatmap with the counts from the confusion matrix, formatted as integers (fmt="d") for clarity in the dashboard.
            cmap="Blues", # Use the "Blues" colormap for the heatmap to visually represent the counts in the confusion matrix, with higher counts appearing in darker shades of blue, making it easier to interpret the performance of each model in the dashboard.
            xticklabels=["Legitimate", "Malicious"], # Set the x-tick labels to "Legitimate" and "Malicious" to indicate the predicted classes in the confusion matrix, enhancing the readability and interpretability of the heatmap in the dashboard.
            yticklabels=["Legitimate", "Malicious"], # Set the y-tick labels to "Legitimate" and "Malicious" to indicate the actual classes in the confusion matrix, enhancing the readability and interpretability of the heatmap in the dashboard.
            ax=ax_cm, cbar=False, # Display the heatmap in the current subplot (ax_cm) and disable the color bar (cbar=False) for a cleaner look in the dashboard, as the annotated counts in the cells provide sufficient information about the values in the confusion matrix without needing a separate color bar.
            linewidths=0.5, # Add thin lines between the cells of the heatmap to improve visual separation and readability of the confusion matrix in the dashboard, making it easier for viewers to distinguish between the different counts in each cell.
            annot_kws={"size": 11, "weight": "bold"}, # Set the annotation text properties to have a font size of 11 and bold weight, enhancing the visibility and emphasis of the counts in each cell of the confusion matrix heatmap for better readability in the dashboard.
        )
        ax_cm.set_xlabel("Predicted", color="white", fontsize=9) # Set the x-axis label to "Predicted" to indicate that the columns of the confusion matrix represent the predicted classes. The label is colored white for contrast against the dark background, and the font size is set to 9 for better readability in the dashboard.
        ax_cm.set_ylabel("Actual", color="white", fontsize=9) # Set the y-axis label to "Actual" to indicate that the rows of the confusion matrix represent the actual classes. The label is colored white for contrast against the dark background, and the font size is set to 9 for better readability in the dashboard.
        ax_cm.set_title( # Set the title of the confusion matrix subplot to the name of the model, replacing " (depth=10)" with a newline for better readability in the case of the Decision Tree model. The title is colored white and styled with bold weight and a font size of 9 to make it stand out against the dark background of the subplot in the dashboard.
            result["model"].replace(" (depth=10)", "\n(depth=10)"), # Set the title of the confusion matrix subplot to the name of the model, replacing " (depth=10)" with a newline for better readability in the case of the Decision Tree model. The title is colored white and styled with bold weight and a font size of 9 to make it stand out against the dark background of the subplot in the dashboard.
            color="white", fontweight="bold", fontsize=9 
        )
        ax_cm.tick_params(colors="white", labelsize=8) # Set the color of the tick labels and ticks to white for contrast against the dark background, and set the font size to 8 for better readability of the axis labels and ticks in the confusion matrix heatmap of the dashboard.
        for spine in ax_cm.spines.values(): # Set the color of the spines (borders) of the confusion matrix heatmap to a dark gray (#444) to create a subtle border around the plot area that contrasts with the background without being too distracting, contributing to the overall professional appearance of the dashboard.
            spine.set_color("#444") # Set the color of the spines (borders) of the confusion matrix heatmap to a dark gray (#444) to create a subtle border around the plot area that contrasts with the background without being too distracting, contributing to the overall professional appearance of the dashboard.

    # ── Save and show ────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True) # Create the output directory if it does not already exist to ensure that the dashboard image can be saved without errors, allowing for organized storage of the generated dashboard images in the specified output directory.
    save_path = os.path.join(output_dir, f"dashboard__{dataset_name}.png") # Define the file path for saving the dashboard image, using the dataset name in the filename to clearly indicate which dataset's results are being displayed in the dashboard, making it easy to identify and differentiate between dashboards for different datasets in the output directory.
    plt.savefig( # Save the generated dashboard figure to the specified file path with a resolution of 150 DPI, using tight bounding box to minimize extra whitespace around the figure, and setting the face color to match the figure's background color for a consistent appearance in the saved image.
        save_path, dpi=150, # Save the generated dashboard figure to the specified file path with a resolution of 150 DPI, using tight bounding box to minimize extra whitespace around the figure, and setting the face color to match the figure's background color for a consistent appearance in the saved image.
        bbox_inches="tight", # Use tight bounding box to minimize extra whitespace around the figure in the saved image, ensuring that the dashboard is neatly cropped and visually appealing when saved to the output directory.
        facecolor=fig.get_facecolor() # Set the face color of the saved image to match the figure's background color, ensuring that the saved dashboard image has a consistent appearance with the generated figure, especially when using a dark theme for the dashboard.
    )
    log.info("Dashboard saved -> %s", os.path.abspath(save_path)) # Log the absolute file path of the saved dashboard image to provide feedback to the user about where the image has been saved, making it easy for them to locate and access the generated dashboard in the output directory.
    plt.show() # Display the generated dashboard figure on the screen, allowing the user to visually analyze the performance metrics, feature importances, and confusion matrices for the different models in an interactive way before or after saving the image to the output directory.


# --------------------------------------------------------------
# Train and evaluate all 3 models on one dataset
# --------------------------------------------------------------
def train_and_evaluate( # Define a function to train and evaluate multiple models on a given dataset, taking the training and testing data, dataset name, and a list to store all results as input parameters. This function will train each model, evaluate their performance using various metrics, and generate a dashboard to visualize the results for the specified dataset.
    X_train, X_test, y_train, y_test, # The training and testing data for the features (X) and target variable (y), which will be used to train the models and evaluate their performance on the specified dataset.
    dataset_name: str, # The name of the dataset being used, which will be included in the output and the dashboard to clearly indicate which dataset's results are being displayed, making it easier to differentiate between results from different datasets when running the function multiple times.
    all_results: list, # A list to store the results of all models across different datasets, allowing for aggregation and comparison of results when the function is called multiple times for different datasets. Each model's results will be appended to this list in a structured format for later analysis and visualization in the dashboard.
) -> None:

    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}") # Print the name of the dataset being processed to provide context for the results that will be displayed in the console and the dashboard, making it clear to the user which dataset's results are being shown, especially when processing multiple datasets in a loop.
    print(f"  Train : {len(X_train)} rows  |  Test: {len(X_test)} rows") # Print the number of rows in the training and testing sets to provide insight into the size of the data being used for training and evaluation, giving the user an understanding of how much data is available for model training and how many samples are being used to evaluate the model's performance in the console output.
    print(f"  Features: {X_train.shape[1]}") # Print the number of features (columns) in the training data to provide information about the dimensionality of the dataset being used for model training, giving the user an understanding of how many input variables are being considered by the models in the console output.
    print(f"{'='*60}")

    model_results = [] # Initialize an empty list to store the results of each model trained and evaluated on the current dataset. This list will be populated with dictionaries containing the performance metrics and confusion matrix values for each model, which will later be used to generate the dashboard and aggregate results across multiple datasets.

    def evaluate(name, model, X_tr, X_te): # Define a nested function to train and evaluate a single model, taking the model name, the model instance, and the training and testing feature data as input parameters. This function will fit the model to the training data, make predictions on the testing data, calculate performance metrics, and store the results in a structured format for later use in the dashboard and overall results aggregation.
        model.fit(X_tr, y_train) # Fit the model to the training data (X_tr and y_train), allowing it to learn the patterns in the data and prepare for making predictions on the testing set. This step is crucial for training the model and enabling it to make informed predictions based on the features provided in the training data.
        y_pred = model.predict(X_te) # Use the trained model to make predictions on the testing feature data (X_te), generating predicted labels (y_pred) that will be compared against the true labels (y_test) to evaluate the model's performance using various metrics in the subsequent steps.
        cm = confusion_matrix(y_test, y_pred) # Compute the confusion matrix using the true labels (y_test) and the predicted labels (y_pred), resulting in a 2x2 matrix that contains the counts of true negatives, false positives, false negatives, and true positives. This matrix will be used to calculate performance metrics such as accuracy, precision, recall, and F1-score for the model's evaluation in the dashboard and console output.
        
        result = { # Create a dictionary to store the results for the current model, including the dataset name, model name, performance metrics (accuracy, precision, recall, F1-score), and the values from the confusion matrix (true negatives, false positives, false negatives, true positives). This structured format allows for easy aggregation of results across multiple models and datasets, and provides all necessary information for generating the dashboard and analyzing model performance in the console output.
            "dataset":   dataset_name,
            "model":     name,
            "accuracy":  round(accuracy_score(y_test, y_pred),  4), # Calculate the accuracy of the model's predictions by comparing the true labels (y_test) with the predicted labels (y_pred), and round the result to 4 decimal places for better readability in the dashboard and console output. Accuracy represents the proportion of correct predictions made by the model out of all predictions, providing a general measure of the model's performance.
            "precision": round(precision_score( # Calculate the precision of the model's predictions by comparing the true labels (y_test) with the predicted labels (y_pred), and round the result to 4 decimal places for better readability in the dashboard and console output. Precision represents the proportion of true positive predictions out of all positive predictions made by the model, indicating how well the model identifies malicious samples without misclassifying legitimate samples as malicious.
                y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score( # Calculate the recall of the model's predictions by comparing the true labels (y_test) with the predicted labels (y_pred), and round the result to 4 decimal places for better readability in the dashboard and console output. Recall represents the proportion of true positive predictions out of all actual positive samples, indicating how well the model identifies malicious samples without missing them.
                y_test, y_pred, zero_division=0), 4),
            "f1_score":  round(f1_score( # Calculate the F1-score of the model's predictions by comparing the true labels (y_test) with the predicted labels (y_pred), and round the result to 4 decimal places for better readability in the dashboard and console output. The F1-score is the harmonic mean of precision and recall, providing a balanced measure of the model's performance in identifying malicious samples while minimizing false positives and false negatives.
                y_test, y_pred, zero_division=0), 4),
            "tn": int(cm[0][0]), "fp": int(cm[0][1]), # Extract the true negatives (tn) and false positives (fp) from the confusion matrix (cm) and store them as integers in the result dictionary. True negatives represent the count of legitimate samples correctly identified as legitimate, while false positives represent the count of legitimate samples incorrectly identified as malicious. These values are important for understanding the model's performance in terms of correctly identifying legitimate samples and avoiding misclassification in the dashboard and console output.
            "fn": int(cm[1][0]), "tp": int(cm[1][1]), # Extract the false negatives (fn) and true positives (tp) from the confusion matrix (cm) and store them as integers in the result dictionary. False negatives represent the count of malicious samples incorrectly identified as legitimate, while true positives represent the count of malicious samples correctly identified as malicious. These values are crucial for understanding the model's performance in terms of correctly identifying malicious samples and minimizing missed detections in the dashboard and console output.
        }
        model_results.append(result) # Append the result dictionary for the current model to the model_results list, which will be used to generate the dashboard and aggregate results across multiple models for the current dataset. This allows for a structured collection of performance metrics and confusion matrix values for each model, facilitating analysis and visualization in the dashboard and console output.
        all_results.append(result) # Append the result dictionary for the current model to the all_results list, which is passed as an argument to the train_and_evaluate function. This allows for aggregation of results across multiple datasets when the function is called multiple times, enabling a comprehensive analysis of model performance across different datasets in the dashboard and console output.

        print(f"\n--- {name} ---")
        print(f"  Accuracy  : {result['accuracy']:.4f}") # Print the accuracy of the model's predictions, formatted to 4 decimal places for better readability in the console output. Accuracy represents the proportion of correct predictions made by the model out of all predictions, providing a general measure of the model's performance.
        print(f"  Precision : {result['precision']:.4f}") # Print the precision of the model's predictions, formatted to 4 decimal places for better readability in the console output. Precision represents the proportion of true positive predictions out of all positive predictions made by the model, indicating how well the model identifies malicious samples without misclassifying legitimate samples as malicious.
        print(f"  Recall    : {result['recall']:.4f}") # Print the recall of the model's predictions, formatted to 4 decimal places for better readability in the console output. Recall represents the proportion of true positive predictions out of all actual positive samples, indicating how well the model identifies malicious samples without missing them.
        print(f"  F1-score  : {result['f1_score']:.4f}") # Print the F1-score of the model's predictions, formatted to 4 decimal places for better readability in the console output. The F1-score is the harmonic mean of precision and recall, providing a balanced measure of the model's performance in identifying malicious samples while minimizing false positives and false negatives.
        print(classification_report( # Print the classification report for the model's predictions, which includes precision, recall, F1-score, and support for each class (legitimate and malicious). The report is formatted to show the metrics for both classes, providing a detailed breakdown of the model's performance in identifying legitimate and malicious samples in the console output.
            y_test, y_pred, # Generate the classification report by comparing the true labels (y_test) with the predicted labels (y_pred), providing a detailed breakdown of precision, recall, F1-score, and support for each class (legitimate and malicious) in the console output for better analysis of the model's performance.
            target_names=["Legitimate", "Malicious"], # Set the target names for the classification report to "Legitimate" and "Malicious" to clearly indicate which metrics correspond to which class in the report, enhancing readability and interpretation of the model's performance in the console output.
            zero_division=0 # Handle cases where there are no positive predictions to avoid division by zero errors in the classification report, ensuring that the report can be generated even if the model fails to predict any malicious samples, which is important for understanding the model's performance in edge cases in the console output.
        ))
        return model # Return the trained model instance, which can be used for further analysis or saving for later use in the dashboard and console output.

    # Model 1: Logistic Regression (needs feature scaling)
    scaler  = StandardScaler() # Initialize a StandardScaler to perform feature scaling on the training and testing data for the Logistic Regression model. Feature scaling is important for algorithms like Logistic Regression that are sensitive to the scale of the input features, ensuring that all features contribute equally to the model's learning process and improving convergence during training in the dashboard and console output.
    X_tr_sc = scaler.fit_transform(X_train) # Fit the StandardScaler to the training data (X_train) and transform it to create a scaled version of the training features (X_tr_sc). This step standardizes the features by removing the mean and scaling to unit variance, which is important for the performance of the Logistic Regression model in the dashboard and console output.
    X_te_sc = scaler.transform(X_test) # Use the fitted StandardScaler to transform the testing data (X_test) to create a scaled version of the testing features (X_te_sc). This ensures that the same scaling applied to the training data is also applied to the testing data, maintaining consistency in feature scaling for the Logistic Regression model's evaluation in the dashboard and console output.

    lr = evaluate( # Train and evaluate the Logistic Regression model using the scaled training and testing data, and store the results in the model_results list for later use in the dashboard and overall results aggregation. The Logistic Regression model is configured with a maximum of 1000 iterations, a random state for reproducibility, and balanced class weights to handle any class imbalance in the dataset, which can improve the model's performance in identifying malicious samples while minimizing false positives in the dashboard and console output.
        "Logistic Regression",
        LogisticRegression( # Initialize the Logistic Regression model with specified parameters for training. The max_iter parameter is set to 1000 to allow for sufficient iterations during training, random_state is set to 42 for reproducibility of results, and class_weight is set to "balanced" to automatically adjust weights inversely proportional to class frequencies
            max_iter=1000, random_state=42,
            class_weight="balanced"
        ),
        X_tr_sc, X_te_sc, # Use the scaled training and testing data for the Logistic Regression model, as this algorithm is sensitive to the scale of the input features. Scaling ensures that all features contribute equally to the model's learning process and can improve convergence during training, leading to better performance in identifying malicious samples in the dashboard and console output.
    )

    # Model 2: Random Forest
    rf = evaluate( # Train and evaluate the Random Forest model using the original (unscaled) training and testing data, and store the results in the model_results list for later use in the dashboard and overall results aggregation. The Random Forest model is configured with 200 estimators (trees), a random state for reproducibility, parallel processing with n_jobs=-1 to utilize all available CPU cores, and balanced class weights to handle any class imbalance in the dataset, which can improve the model's performance in identifying malicious samples while minimizing false positives in the dashboard and console output.
        "Random Forest",
        RandomForestClassifier( # Initialize the Random Forest model with specified parameters for training. The n_estimators parameter is set to 200 to specify the number of trees in the forest, random_state is set to 42 for reproducibility of results, n_jobs is set to -1 to utilize all available CPU cores for parallel processing during training, and class_weight is set to "balanced" to automatically adjust weights inversely proportional to class frequencies, which can improve the model's performance in identifying malicious samples while minimizing false positives in the dashboard and console output.
            n_estimators=200, random_state=42,
            n_jobs=-1, class_weight="balanced"
        ),
        X_train, X_test, # Use the original (unscaled) training and testing data for the Random Forest model, as this algorithm is not sensitive to the scale of the input features. Random Forest can handle features of varying scales without the need for feature scaling, allowing it to effectively learn from the data and identify malicious samples in the dashboard and console output without the additional step of scaling the features.
    )

    importances = pd.Series( # Create a Pandas Series to store the feature importances from the Random Forest model, using the feature names from the training data (X_train.columns) as the index. This Series will be sorted in descending order to identify the most important features that contributed to the model's predictions, which can provide insights into which features are most influential in identifying malicious samples in the dashboard and console output.
        rf.feature_importances_, index=X_train.columns 
    ).sort_values(ascending=False) # Sort the feature importances in descending order to identify the most important features that contributed to the Random Forest model's predictions. This allows for a clear visualization of the top features in the dashboard and provides insights into which features are most influential in identifying malicious samples in the console output.

    print("\n  Top 10 Feature Importances (Random Forest):") # Print a header for the top 10 feature importances from the Random Forest model to provide context for the list of features and their importance scores that will be displayed in the console output.
    for feat, score in importances.head(10).items(): # Iterate through the top 10 feature importances from the Random Forest model and print each feature name along with its corresponding importance score, formatted to 4 decimal places for better readability in the console output. This allows the user to see which features were most influential in the model's predictions and can provide insights into the factors that contribute to identifying malicious samples in the dashboard and console output.
        print(f"    {feat:<35s} {score:.4f}") # Print the feature name (left-aligned with a width of 35 characters) and its corresponding importance score (formatted to 4 decimal places) for each of the top 10 features from the Random Forest model, providing a clear and organized display of the most influential features in the console output for better analysis and interpretation of the model's performance in identifying malicious samples in the dashboard and console output.

    # Model 3: Decision Tree
    dt = evaluate( # Train and evaluate the Decision Tree model using the original (unscaled) training and testing data, and store the results in the model_results list for later use in the dashboard and overall results aggregation. The Decision Tree model is configured with a maximum depth of 10 to prevent overfitting, a minimum samples split of 20 to ensure that nodes are not split if they contain fewer than 20 samples, a random state for reproducibility, and balanced class weights to handle any class imbalance in the dataset, which can improve the model's performance in identifying malicious samples while minimizing false positives in the dashboard and console output.
        "Decision Tree (depth=10)", 
        DecisionTreeClassifier( # Initialize the Decision Tree model with specified parameters for training. The max_depth parameter is set to 10 to limit the depth of the tree and prevent overfitting, min_samples_split is set to 20 to ensure that nodes are not split if they contain fewer than 20 samples, random_state is set to 42 for reproducibility of results, and class_weight is set to "balanced" to automatically adjust weights inversely proportional to class frequencies
            max_depth=10, min_samples_split=20,
            random_state=42, class_weight="balanced"
        ),
        X_train, X_test, # Use the original (unscaled) training and testing data for the Decision Tree model, as this algorithm is not sensitive to the scale of the input features. Decision Trees can handle features of varying scales without the need for feature scaling, allowing it to effectively learn from the data and identify malicious samples in the dashboard and console output without the additional step of scaling the features.
    )

    # Save trained model artifacts for use by predict.py

    log.info("Saving trained model artifacts to %s/", OUTPUT_DIR) # Log the action of saving the trained model artifacts to the specified output directory, providing feedback to the user that the models and scaler are being saved for later use in the predict.py script, which will allow for making predictions on new data using the trained models in the dashboard and console output.

    with open(os.path.join(OUTPUT_DIR, "model_lr.pkl"), "wb") as f: # Open a file in the output directory named "model_lr.pkl" in binary write mode to save the trained Logistic Regression model using pickle. This allows the model to be saved in a serialized format that can be loaded later for making predictions on new data in the predict.py script, enabling the use of the trained Logistic Regression model in the dashboard and console output for future predictions.
        pickle.dump(lr, f) # Use pickle to serialize and save the trained Logistic Regression model (lr) to the file "model_lr.pkl" in the output directory, allowing it to be loaded later for making predictions on new data in the predict.py script, enabling the use of the trained Logistic Regression model in the dashboard and console output for future predictions.

    with open(os.path.join(OUTPUT_DIR, "model_rf.pkl"), "wb") as f: # Open a file in the output directory named "model_rf.pkl" in binary write mode to save the trained Random Forest model using pickle. This allows the model to be saved in a serialized format that can be loaded later for making predictions on new data in the predict.py script, enabling the use of the trained Random Forest model in the dashboard and console output for future predictions.
        pickle.dump(rf, f) # Use pickle to serialize and save the trained Random Forest model (rf) to the file "model_rf.pkl" in the output directory, allowing it to be loaded later for making predictions on new data in the predict.py script, enabling the use of the trained Random Forest model in the dashboard and console output for future predictions.

    with open(os.path.join(OUTPUT_DIR, "model_dt.pkl"), "wb") as f: # Open a file in the output directory named "model_dt.pkl" in binary write mode to save the trained Decision Tree model using pickle. This allows the model to be saved in a serialized format that can be loaded later for making predictions on new data in the predict.py script, enabling the use of the trained Decision Tree model in the dashboard and console output for future predictions.
        pickle.dump(dt, f) # Use pickle to serialize and save the trained Decision Tree model (dt) to the file "model_dt.pkl" in the output directory, allowing it to be loaded later for making predictions on new data in the predict.py script, enabling the use of the trained Decision Tree model in the dashboard and console output for future predictions.

    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f: # Open a file in the output directory named "scaler.pkl" in binary write mode to save the fitted StandardScaler using pickle. This allows the scaler to be saved in a serialized format that can be loaded later for scaling new data in the predict.py script, ensuring that the same feature scaling applied during training is also applied to new data when making predictions with the trained Logistic Regression model in the dashboard and console output for future predictions.
        pickle.dump(scaler, f) # Use pickle to serialize and save the fitted StandardScaler (scaler) to the file "scaler.pkl" in the output directory, allowing it to be loaded later for scaling new data in the predict.py script, ensuring that the same feature scaling applied during training is also applied to new data when making predictions with the trained Logistic Regression model in the dashboard and console output for future predictions.

    log.info("Model artifacts saved — predict.py is now operational.") # Log a message indicating that the model artifacts have been saved and that the predict.py script is now operational, providing feedback to the user that the trained models and scaler are available for making predictions on new data in the dashboard and console output for future predictions.

    # 5-Fold Cross-Validation
    print(f"\n  5-Fold Cross-Validation F1 (training set):") # Print a header for the 5-fold cross-validation F1 scores on the training set to provide context for the cross-validation results that will be displayed in the console output, allowing the user to understand that the following scores represent the model's performance on the training data using cross-validation.
    for name, model, X_cv in [ # Iterate through the list of models (Logistic Regression, Random Forest, Decision Tree) along with their corresponding training data (scaled for Logistic Regression and original for the others) to perform 5-fold cross-validation and evaluate their F1 scores on the training set. This allows for an assessment of the model's performance and generalization ability on the training data, providing insights into how well each model is likely to perform on unseen data in the dashboard and console output.
        ("Logistic Regression",
         LogisticRegression( # Initialize the Logistic Regression model with specified parameters for cross-validation.
             max_iter=1000, random_state=42, # Set the maximum number of iterations to 1000 to allow for sufficient training time for the Logistic Regression model, and set the random state to 42 for reproducibility of results across different runs. This ensures that the model's training process is consistent and can be replicated, which is important for comparing results and analyzing performance in the dashboard and console output.
             class_weight="balanced"), # Initialize the Logistic Regression model with specified parameters for cross-validation. The max_iter parameter is set to 1000 to allow for sufficient iterations during training, random_state is set to 42 for reproducibility of results, and class_weight is set to "balanced" to automatically adjust weights inversely proportional to class frequencies
         X_tr_sc), # Use the scaled training data (X_tr_sc) for the Logistic Regression model during cross-validation, as this algorithm is sensitive to the scale of the input features. Scaling ensures that all features contribute equally to the model's learning process and can improve convergence during training, leading to better performance in identifying malicious samples in the dashboard and console output.
        ("Random Forest", # Initialize the Random Forest model with specified parameters for cross-validation.
         RandomForestClassifier( 
             n_estimators=100, random_state=42,
             class_weight="balanced"), # Specified parameters for cross-validation. The n_estimators parameter is set to 100 to specify the number of trees in the forest for cross-validation, random_state is set to 42 for reproducibility of results, and class_weight is set to "balanced" to automatically adjust weights inversely proportional to class frequencies, which can improve the model's performance in identifying malicious samples while minimizing false positives in the dashboard and console output.
         X_train),
        ("Decision Tree",
         DecisionTreeClassifier( 
             max_depth=10, random_state=42,
             class_weight="balanced"), # Initialize the Decision Tree model with specified parameters for cross-validation. The max_depth parameter is set to 10 to limit the depth of the tree and prevent overfitting during cross-validation, random_state is set to 42 for reproducibility of results, and class_weight is set to "balanced" to automatically adjust weights inversely proportional to class frequencies
         X_train), # Use the original (unscaled) training data (X_train) for the Random Forest and Decision Tree models during cross-validation, as these algorithms are not sensitive to the scale of the input features. They can handle features of varying scales without the need for feature scaling, allowing them to effectively learn from the data and identify malicious samples in the dashboard and console output without the additional step of scaling the features.
    ]:
        cv = cross_val_score( # Perform 5-fold cross-validation on the training data (X_cv and y_train) using the specified model, and evaluate the F1 score for each fold. The StratifiedKFold is used to ensure that each fold has a representative distribution of the classes, which is important for imbalanced datasets. The scoring parameter is set to "f1" to evaluate the model's performance based on the F1 score, which is a balanced measure of precision and recall, providing insights into how well the model identifies malicious samples while minimizing false positives and false negatives in the dashboard and console output.
            model, X_cv, y_train, # Use the specified model and training data for cross-validation, allowing for an assessment of the model's performance and generalization ability on the training data, providing insights into how well each model is likely to perform on unseen data in the dashboard and console output.
            cv=StratifiedKFold( 
                n_splits=5, shuffle=True, random_state=42), # Use StratifiedKFold with 5 splits, shuffling, and a random state for reproducibility to ensure that each fold has a representative distribution of the classes during cross-validation, which is important for imbalanced datasets. This allows for a more accurate assessment of the model's performance in identifying malicious samples while minimizing false positives and false negatives in the dashboard and console output.
            scoring="f1", n_jobs=-1, # Set the scoring parameter to "f1" to evaluate the model's performance based on the F1 score during cross-validation, which is a balanced measure of precision and recall, providing insights into how well the model identifies malicious samples while minimizing false positives and false negatives in the dashboard and console output. The n_jobs parameter is set to -1 to utilize all available CPU cores for parallel processing during cross-validation, which can speed up the evaluation process, especially for larger datasets and more complex models in the dashboard and console output.
        )
        print(f"    {name:<25s} mean={cv.mean():.4f}  " # Print the name of the model (left-aligned with a width of 25 characters) along with the mean F1 score from the cross-validation, formatted to 4 decimal places for better readability in the console output. This provides a summary of the model's performance on the training data using cross-validation, allowing the user to compare the models' ability to identify malicious samples while minimizing false positives and false negatives in the dashboard and console output.
              f"std={cv.std():.4f}") # Print the standard deviation of the F1 scores from the cross-validation, formatted to 4 decimal places for better readability in the console output. This provides insight into the variability of the model's performance across different folds, allowing the user to understand how consistent the model's performance is in identifying malicious samples while minimizing false positives and false negatives in the dashboard and console output.

    show_dashboard( # Call the show_dashboard function to generate and display a visual dashboard of the model results and feature importances for the current dataset. The dashboard will provide a visual representation of the performance metrics (accuracy, precision, recall, F1-score) for each model, as well as a visualization of the top feature importances from the Random Forest model, allowing for an easy comparison of model performance and insights into which features are most influential in identifying malicious samples in the dashboard and console output.
        dataset_name, model_results, importances, OUTPUT_DIR # Pass the dataset
    )


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main(test_size: float = 0.2) -> None: # Define the main function that orchestrates the loading of datasets, training and evaluation of models, and generation of results and dashboards. The test_size parameter allows for specifying the fraction of the dataset to be used as the test set during the train-test split, providing flexibility in how the data is partitioned for model training and evaluation in the dashboard and console output.
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Create the output directory if it does not already exist, ensuring that there is a designated location for saving the trained model artifacts, results, and dashboards generated during the execution of the main function. This allows for organized storage of outputs and easy access to the results in the dashboard and console output.
    log_path    = setup_logging(OUTPUT_DIR) # Set up logging to capture the output of the main function and save it to a log file in the output directory. This allows for tracking the execution of the main function, including any errors or important information, and provides a record of the results and performance metrics for later analysis in the dashboard and console output.
    all_results = [] # Initialize an empty list to store the results of model performance across all datasets. This list will be populated with dictionaries containing the performance metrics and confusion matrix values for each model and dataset, which will later be used to generate a comprehensive summary of results across multiple datasets in the dashboard and console output.

    for feature_path in FEATURE_FILES: # Iterate through the list of feature file paths specified in FEATURE_FILES to load each dataset, train and evaluate the models, and generate results and dashboards for each dataset. This allows for a systematic processing of multiple datasets, enabling a comprehensive analysis of model performance across different datasets in the dashboard and console output.
        if not os.path.exists(feature_path): # Check if the feature file exists at the specified path, and if it does not, log a warning message indicating that the file was not found and skip to the next iteration of the loop. This ensures that the main function can continue processing other datasets even if one of the feature files is missing, providing robustness in handling potential issues with dataset availability in the dashboard and console output.
            log.warning( # Log a warning message indicating that the feature file was not found at the specified path, providing feedback to the user about the missing file and allowing them to take corrective action if necessary. This helps to ensure that the user is aware of any issues with dataset availability and can address them to enable successful execution of the main function for all datasets in the dashboard and console output.
                "Feature file not found, skipping: %s", feature_path
            )
            continue # Skip to the next iteration of the loop if the feature file is not found, allowing the main function to continue processing other datasets without interruption, providing robustness in handling potential issues with dataset availability in the dashboard and console output.

        log.info("Loading: %s", feature_path) # Log an informational message indicating that the dataset is being loaded from the specified feature file path, providing feedback to the user about the progress of the main function and allowing them to track which dataset is currently being processed for model training and evaluation in the dashboard and console output.
        df = load_dataset(feature_path) # Load the dataset from the specified feature file path using the load_dataset function, which reads the data into a Pandas DataFrame. This allows for easy manipulation and analysis of the dataset, enabling the subsequent steps of identifying the label column, normalizing labels, and preparing the data for model training and evaluation in the dashboard and console output.

        label_col = find_label_column(df) # Use the find_label_column function to identify the name of the label column in the loaded dataset DataFrame (df). This function checks for common label column names such as "label", "class", "target",
        if label_col is None: # If the find_label_column function returns None, indicating that no label column was found in the dataset, log an error message indicating that the label column was not found in the specified feature file path and skip to the next iteration of the loop. This ensures that the main function can continue processing other datasets even if one of them does not have a recognizable label column, providing robustness in handling potential issues with dataset structure in the dashboard and console output.
            log.error("No label column found in %s", feature_path) # Log an error message indicating that no label column was found in the specified feature file path, providing feedback to the user about the issue with the dataset structure and allowing them to take corrective action if necessary. This helps to ensure that the user is aware of any issues with the dataset structure and can address them to enable successful execution of the main function for all datasets in the dashboard and console output.
            continue # Skip to the next iteration of the loop if no label column is found, allowing the main function to continue processing other datasets without interruption, providing robustness in handling potential issues with dataset structure in the dashboard and console output.

        y    = normalize_labels(df[label_col]) # Use the normalize_labels function to convert the values in the identified label column (df[label_col]) to a standardized format of 0 for legitimate samples and 1 for malicious samples. This function handles various representations of labels (e.g., "legitimate", "malicious", "benign", "phishing") and ensures that the labels are consistent across different datasets, which is important for training and evaluating the models in the dashboard and console output.
        mask = y.isin([0, 1]) # Create a boolean mask to filter the dataset to include only rows where the normalized labels are either 0 (legitimate) or 1 (malicious). This ensures that only valid samples with recognizable labels are included in the model training and evaluation process, providing a clean and consistent dataset for analysis in the dashboard and console output.
        df   = df[mask].reset_index(drop=True) # Filter the original DataFrame (df) using the boolean mask to include only valid rows with normalized labels of 0 or 1, and reset the index of the resulting DataFrame to ensure a clean and sequential index for the filtered dataset. This allows for a consistent and organized dataset for model training and evaluation in the dashboard and console output.
        y    = y[mask].reset_index(drop=True).astype(int) # Apply the same boolean mask to the normalized labels (y) to filter it to include only valid labels corresponding to the filtered dataset, reset the index of the resulting Series, and convert the labels to integers (0 and 1) for use in model training and evaluation. This ensures that the labels are consistent with the filtered dataset and are in the correct format for training the models in the dashboard and console output.

        if len(df) == 0: # Check if the resulting DataFrame (df) is empty after filtering for valid labels, and if it is, log an error message indicating that there are no valid rows in the specified feature file path after cleaning and skip to the next iteration of the loop. This ensures that the main function can continue processing other datasets even if one of them does not contain any valid samples after cleaning, providing robustness in handling potential issues with dataset quality in the dashboard and console output.
            log.error("No valid rows in %s after cleaning.", feature_path) # Log an error message indicating that there are no valid rows in the specified feature file path after cleaning, providing feedback to the user about the issue with the dataset quality and allowing them to take corrective action if necessary. This helps to ensure that the user is aware of any issues with the dataset quality and can address them to enable successful execution of the main function for all datasets in the dashboard and console output.
            continue # Skip to the next iteration of the loop if there are no valid rows in the dataset after cleaning, allowing the main function to continue processing other datasets without interruption, providing robustness in handling potential issues with dataset quality in the dashboard and console output.

        feature_cols = [ # Create a list of feature column names by including all columns in the DataFrame (df) except for the label column and the "url" column. This ensures that only relevant features are included in the model training and evaluation process, while excluding the label and any non-feature columns that may not be useful for modeling in the dashboard and console output.
            c for c in df.columns # Iterate through all column names in the DataFrame (df) and include only those that are not the label column (label_col) and not the "url" column, which is likely to contain non-numeric data that is not useful for modeling. This allows for a clean set of feature columns to be used for training the models in the dashboard and console output.
            if c not in (label_col, "url") # Exclude the label column (label_col) and the "url" column from the list of feature columns, as the label column is used for training the models and should not be included as a feature, and the "url" column likely contains non-numeric data that is not useful for modeling. This ensures that only relevant numeric features are included in the model training and evaluation process in the dashboard and console output.
        ]
        X = df[feature_cols].select_dtypes(include=["number"]) # Create a new DataFrame (X) that includes only the feature columns from the original DataFrame (df) that are of numeric data types. This ensures that only numeric features are included in the model training and evaluation process, as most machine learning algorithms require numeric input. This also helps to avoid issues with non-numeric data that may not be suitable for modeling in the dashboard and console output.

        log.info( # Log an informational message summarizing the dataset being processed, including the number of rows, number of features, and the distribution of malicious and legitimate samples in terms of count and percentage. This provides a clear overview of the dataset's characteristics and class distribution, which is important for understanding the context of the model training and evaluation results in the dashboard and console output.
            "%d rows, %d features | malicious: %d (%.1f%%)  " # Log the number of rows and features in the dataset, as well as the count and percentage of malicious and legitimate samples. This information is crucial for understanding the dataset's characteristics and class distribution, which can impact model performance and the interpretation of results in the dashboard and console output.
            "legitimate: %d (%.1f%%)",
            len(X), len(X.columns), # Log the number of rows and features in the dataset, which provides insight into the size and complexity of the dataset being processed for model training and evaluation in the dashboard and console output.
            int(y.sum()), 100 * y.mean(), # Log the count and percentage of malicious samples in the dataset, which is calculated by summing the values in the label series (y) to get the count of malicious samples, and multiplying the mean of the label series by 100 to get the percentage of malicious samples. This information is important for understanding the class distribution in the dataset, which can impact model performance and the interpretation of results in the dashboard and console output.
            int((y == 0).sum()), 100 * (1 - y.mean()), # Log the count and percentage of legitimate samples in the dataset, which is calculated by counting the number of samples where the label is 0 (legitimate) using (y == 0).sum(), and calculating the percentage of legitimate samples as 100 times (1 - y.mean()), since y.mean() gives the percentage of malicious samples. This information is important for understanding the class distribution in the dataset, which can impact model performance and the interpretation of results in the dashboard and console output.
        )

        X_train, X_test, y_train, y_test = train_test_split( # Split the dataset into training and testing sets using the train_test_split function from scikit-learn. The test_size parameter specifies the fraction of the dataset to be used as the test set, random_state is set to 42 for reproducibility of results, and stratify=y ensures that the class distribution is preserved in both the training and testing sets. This allows for a fair evaluation of the models on unseen data while maintaining the class distribution, which is important for interpreting the results in the dashboard and console output.
            X, y, # Split the feature DataFrame (X) and label series (y) into training and testing sets, allowing for the training of models on the training data and evaluation of their performance on the unseen test data in the dashboard and console output.
            test_size=test_size, # Specify the fraction of the dataset to be used as the test set, which allows for flexibility in how the data is partitioned for model training and evaluation in the dashboard and console output. A common default value is 0.2 (20% test set), but this can be adjusted based on the size of the dataset and the desired balance between training and testing data.
            random_state=42, # Set the random state to 42 for reproducibility of results, ensuring that the same train-test split is generated each time the main function is executed. This allows for consistent evaluation of model performance across different runs and facilitates comparison of results in the dashboard and console output.
            stratify=y, # Use stratification based on the label series (y) to ensure that the class distribution is preserved in both the training and testing sets. This is important for imbalanced datasets, as it helps to ensure that both sets contain a representative distribution of malicious and legitimate samples, which can impact model performance and the interpretation of results in the dashboard and console output.
        )

        dataset_name = os.path.splitext( # Extract the dataset name from the feature file path by taking the base name of the file (without the directory path) and removing the file extension. This provides a clean and concise name for the dataset that can be used in the dashboard and console output to identify which dataset is being processed and evaluated.
            os.path.basename(feature_path) # Get the base name of the feature file path, which is the file name without the directory path. This allows for a cleaner and more concise dataset name to be used in the dashboard and console output, making it easier to identify which dataset is being processed and evaluated.
        )[0] # Remove the file extension from the base name to get the dataset name, which can be used in the dashboard and console output to identify which dataset is being processed and evaluated. This allows for a clearer presentation of results and insights specific to each dataset in the dashboard and console output.

        train_and_evaluate( # Call the train_and_evaluate function to train and evaluate the models on the current dataset, passing the training and testing data, dataset name, and the all_results list to store the results for later aggregation. This function will handle the training of the Logistic Regression, Random Forest, and Decision Tree models, as well as the generation of performance metrics and feature importances, which will be used to create a visual dashboard and provide insights in the console output for each dataset in the dashboard and console output.
            X_train, X_test, # Pass the training and testing feature data (X_train and X_test) to the train_and_evaluate function, which will use this data to train the models and evaluate their performance on the test set. This allows for an assessment of how well the models generalize to unseen data, providing insights into their performance in identifying malicious samples in the dashboard and console output.
            y_train, y_test, # Pass the training and testing label data (y_train and y_test) to the train_and_evaluate function, which will use this data to train the models and evaluate their performance on the test set. This allows for an assessment of how well the models generalize to unseen data, providing insights into their performance in identifying malicious samples in the dashboard and console output.
            dataset_name, all_results, # Pass the dataset
        )

    if all_results: # After processing all datasets, check if there are any results stored in the all_results list. If there are results, create a DataFrame from the list of results and save it to a CSV file in the output directory. This allows for a comprehensive summary of model performance across all datasets to be saved and easily accessed for further analysis in the dashboard and console output.
        results_df   = pd.DataFrame(all_results) # Create a DataFrame from the list of results stored in all_results, which contains dictionaries of performance metrics and confusion matrix values for each model and dataset. This DataFrame will provide a structured format for analyzing and comparing the results across different datasets in the dashboard and console output.
        results_path = os.path.join(OUTPUT_DIR, "model_results.csv") # Define the path for saving the results CSV file in the output directory, which will contain the comprehensive summary of model performance across all datasets. This allows for easy access to the results for further analysis and reference in the dashboard and console output.
        results_df.to_csv(results_path, index=False) # Save the results DataFrame to a CSV file at the specified path, without including the index in the CSV file. This allows for a clean and organized presentation of the results, making it easier to analyze and compare model performance across different datasets in the dashboard and console output.
        log.info( # Log an informational message indicating that all results have been saved to the specified path, providing feedback to the user about the successful saving of the results and allowing them to access the comprehensive summary of model performance across all datasets in the dashboard and console output.
            "All results saved -> %s",
            os.path.abspath(results_path) # Log the absolute path to the saved results CSV file, providing clear information to the user about where the comprehensive summary of model performance across all datasets can be accessed for further analysis in the dashboard and console output.
        )

        print(f"\n{'='*60}")
        print("FULL RESULTS SUMMARY") # Print a header for the full results summary in the console output, providing context for the comprehensive summary of model performance across all datasets that will be displayed. This allows the user to understand that the following information represents a summary of results from all processed datasets in the dashboard and console output.
        print(f"{'='*60}")
        print(results_df.to_string(index=False)) # Print the full results DataFrame as a string in the console output, without including the index. This provides a comprehensive summary of model performance across all datasets in a readable format, allowing the user to analyze and compare the results for each model and dataset in the dashboard and console output.

    print("\nTraining complete for all datasets.") # Print a message indicating that the training process is complete for all datasets, providing feedback to the user about the completion of the main function and allowing them to know that all datasets have been processed and evaluated in the dashboard and console output.

    # Close the tee and confirm where the results file was saved
    if isinstance(sys.stdout, _Tee): # Check if sys.stdout is an instance of the _Tee class, which indicates that the output is being captured and saved to a log file. If it is, close the sys.stdout to ensure that the log file is properly closed and the output is saved correctly, and then reset sys.stdout to its original value (sys.__stdout__) to restore normal console output. This allows for proper handling of the logging mechanism and ensures that the results are saved to the log file while also allowing for normal console output after the training process is complete in the dashboard and console output.
        sys.stdout.close() # Close the sys.stdout to ensure that the log file is properly closed and the output is saved correctly, which is important for ensuring that all logged information is written to the file and that there are no issues with file handling in the dashboard and console output.
        sys.stdout = sys.__stdout__ # Reset sys.stdout to its original value (sys.__stdout__) to restore normal console output, allowing for regular printing to the console after the training process is complete and the results have been saved to the log file. This ensures that any further print statements will be displayed in the console as expected, providing a seamless user experience in the dashboard and console output.
    print(f"\nFull results saved -> {os.path.abspath(log_path)}") # Print a message indicating the absolute path to the saved log file, providing clear information to the user about where the full results of the training process, including any logged information and performance metrics, can be accessed for further analysis in the dashboard and console output.


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------
def parse_args() -> argparse.Namespace: # Define a function to parse command-line arguments using the argparse library. This function will allow the user to specify the test set fraction for the train-test split when executing the script, providing flexibility in how the data is partitioned for model training and evaluation in the dashboard and console output.
    p = argparse.ArgumentParser( # Create an ArgumentParser object to handle command-line arguments, providing a description of the script's functionality for better user understanding when using the --help option in the console output.
        description="Train 3 ML models and display visual results dashboards." # Set the description of the script to provide context for the user about what the script does, which is to train three machine learning models (Logistic Regression, Random Forest, Decision Tree) and display visual results dashboards for each dataset processed. This helps users understand the purpose of the script and what to expect from its execution when they run it in the console output.
    )
    p.add_argument( # Add a command-line argument for specifying the test set fraction for the train-test split, allowing the user to customize how the data is partitioned for model training and evaluation in the dashboard and console output. The argument is optional and defaults to 0.2 (20% test set) if not provided.
        "--testsize", type=float, default=0.2, # Add an optional command-line argument named --testsize that allows the user to specify the fraction of the dataset to be used as the test set during the train-test split. The type of the argument is set to float, and it defaults to 0.2 (20% test set) if not provided by the user. This provides flexibility in how the data is partitioned for model training and evaluation in the dashboard and console output, allowing users to adjust the test set size based on their specific needs and preferences when running the script in the console output.
        help="Test set fraction (default: 0.2)" # Set the help message for the --testsize argument to provide guidance to the user about what the argument does and its default value when using the --help option in the console output. This helps users understand how to use the argument and what it controls in the script, which is the fraction of the dataset used as the test set during the train-test split for model training and evaluation in the dashboard and console output.
    )
    return p.parse_args() # Parse the command-line arguments and return them as a Namespace object, allowing the main function to access the specified test set fraction for the train-test split when executing the script. This enables users to customize the data partitioning for model training and evaluation in the dashboard and console output based on their specific needs and preferences when running the script in the console output.


if __name__ == "__main__": # Check if the script is being run as the main program, and if so, execute the following code block. This allows for the main function to be called when the script is executed directly, while also allowing for the functions defined in the script to be imported and used in other contexts without executing the main function, providing flexibility in how the code can be utilized in different scenarios in the dashboard and console output.
    try: # Use a try-except block to catch any unhandled exceptions that may occur during the execution of the main function. If an exception is caught, log an error message with the traceback information to provide insights into what went wrong, and then exit the program with a non-zero status code to indicate that an error occurred. This helps to ensure that any issues during execution are properly logged and that the user is informed about the error in a clear and informative manner in the dashboard and console output.
        args = parse_args() # Call the parse_args function to parse command-line arguments and store them in the args variable, allowing the main function to access the specified test set fraction for the train-test split when executing the script. This enables users to customize the data partitioning for model training and evaluation in the dashboard and console output based on their specific needs and preferences when running the script in the console output.
        main(test_size=args.testsize) # Call the main function with the specified test set fraction for the train-test split, which is obtained from the parsed command-line arguments (args.testsize). This allows for the execution of the main function with a customizable test set size, enabling users to adjust how the data is partitioned for model training and evaluation in the dashboard and console output based on their specific needs and preferences when running the script in the console output.
    except Exception: # Catch any unhandled exceptions that occur during the execution of the main function, log an error message with the traceback information to provide insights into what went wrong, and then exit the program with a non-zero status code to indicate that an error occurred. This helps to ensure that any issues during execution are properly logged and that the user is informed about the error in a clear and informative manner in the dashboard and console output.
        log.error("Unhandled error:\n%s", traceback.format_exc()) # Log an error message with the traceback information of the unhandled exception, providing insights into what went wrong during the execution of the main function. This helps to ensure that any issues are properly logged and that the user is informed about the error in a clear and informative manner in the dashboard and console output.
        sys.exit(1) # Exit the program with a non-zero status code (1) to indicate that an error occurred during the execution of the main function. This allows for proper handling of errors and provides feedback to the user about the issue, while also preventing further execution of the program in an erroneous state in the dashboard and console output.