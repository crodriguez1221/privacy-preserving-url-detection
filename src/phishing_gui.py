# -*- coding: utf-8 -*-
# --------------------------------------------------------------
# src/phishing_gui.py
# --------------------------------------------------------------
# Purpose:
#   A graphical user interface (GUI) for the phishing detection
#   system. Allows users to classify URLs as Malicious or
#   Legitimate without using the command line.
#
#   This GUI wraps the same feature engineering and model
#   inference logic as predict.py, providing a visual interface
#   suitable for demonstrations and presentations.
#
#   Uses tkinter — part of the Python standard library.
#   No additional packages need to be installed.
#
# Prerequisites:
#   train_models.py must have been run first so that the
#   following .pkl files exist in outputs/:
#       outputs/model_lr.pkl
#       outputs/model_rf.pkl
#       outputs/model_dt.pkl
#       outputs/scaler.pkl
#
# Usage:
#   python src/phishing_gui.py
# --------------------------------------------------------------

import os               # for file path handling and checking if .pkl files exist
import sys              # for sys.exit if model files are missing
import pickle           # for loading the serialised .pkl model and scaler files
import tkinter as tk    # the GUI framework — part of Python standard library, no install needed
from tkinter import ttk, messagebox, font  # ttk for styled widgets, messagebox for error popups
from urllib.parse import urlparse          # for decomposing the URL into components for feature extraction

import pandas as pd     # for creating the single-row feature DataFrame the classifiers expect
import numpy as np      # for Shannon entropy calculation


# --------------------------------------------------------------
# File paths — must match those used by train_models.py
# --------------------------------------------------------------
OUTPUT_DIR  = "outputs"                                      # directory where train_models.py saves model artifacts
MODEL_LR    = os.path.join(OUTPUT_DIR, "model_lr.pkl")      # Logistic Regression model artifact
MODEL_RF    = os.path.join(OUTPUT_DIR, "model_rf.pkl")      # Random Forest model artifact
MODEL_DT    = os.path.join(OUTPUT_DIR, "model_dt.pkl")      # Decision Tree model artifact
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")        # StandardScaler fitted on training data


# --------------------------------------------------------------
# Colour scheme — dark professional theme matching the dashboards
# --------------------------------------------------------------
BG_DARK      = "#0D1117"   # main background — matches dashboard dark background
BG_PANEL     = "#161B22"   # panel/card background
BG_INPUT     = "#21262D"   # input field background
ACCENT_BLUE  = "#2E75B6"   # blue accent — matches within-dataset dashboard theme
TEXT_WHITE   = "#E6EDF3"   # primary text colour
TEXT_GREY    = "#8B949E"   # secondary/muted text colour
GREEN        = "#2EA043"   # legitimate verdict colour
RED          = "#DA3633"   # malicious verdict colour
YELLOW       = "#D29922"   # caution/mixed verdict colour
BORDER       = "#30363D"   # border/separator colour


# --------------------------------------------------------------
# Feature engineering — identical to predict.py and extract_features.py
# These functions must stay synchronised with extract_features.py.
# --------------------------------------------------------------
def _shannon_entropy(s: str) -> float:
    """
    Compute character-level Shannon entropy for a string.
    Identical to the function in predict.py and extract_features.py.
    Returns 0.0 for empty strings to avoid log(0) errors.
    """
    if not s:
        return 0.0
    freq = pd.Series(list(s)).value_counts(normalize=True)
    return float(-(freq * np.log2(freq)).sum())


def extract_features_single(url: str) -> pd.DataFrame:
    """
    Engineer all 24 URL-string features from a single URL string.
    Returns a 1-row, 24-column DataFrame matching the training schema.
    Identical logic to predict.py — no network connections made.
    """
    url    = str(url).strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    hostname = parsed.hostname or ""
    path     = parsed.path     or ""
    query    = parsed.query    or ""

    url_length      = len(url)
    hostname_length = len(hostname)
    path_length     = len(path)
    query_length    = len(query)
    num_dots        = url.count(".")
    num_hyphens     = url.count("-")
    num_underscores = url.count("_")
    num_slashes     = url.count("/")
    num_qmarks      = url.count("?")
    num_equals      = url.count("=")
    num_ampersands  = url.count("&")
    num_at          = url.count("@")
    num_percent     = url.count("%")
    num_digits      = sum(c.isdigit() for c in url)
    has_https       = int(parsed.scheme == "https")
    has_ip          = int(bool(
        hostname and
        all(part.isdigit() and 0 <= int(part) <= 255
            for part in hostname.split("."))
        and hostname.count(".") == 3
    ))
    has_port        = int(bool(parsed.port))
    has_www         = int(hostname.startswith("www."))
    has_at_in_url   = int("@" in url)
    subdomain_depth = max(0, hostname.count(".") - 1) if hostname else 0
    url_entropy      = _shannon_entropy(url)
    hostname_entropy = _shannon_entropy(hostname)
    url_len_safe     = url_length if url_length > 0 else np.nan
    digit_ratio      = num_digits / url_len_safe
    special_char_ratio = (
        (num_dots + num_hyphens + num_underscores + num_at +
         num_percent + num_qmarks + num_equals + num_ampersands)
        / url_len_safe
    )

    features = {
        "url_length": url_length, "hostname_length": hostname_length,
        "path_length": path_length, "query_length": query_length,
        "num_dots": num_dots, "num_hyphens": num_hyphens,
        "num_underscores": num_underscores, "num_slashes": num_slashes,
        "num_qmarks": num_qmarks, "num_equals": num_equals,
        "num_ampersands": num_ampersands, "num_at": num_at,
        "num_percent": num_percent, "num_digits": num_digits,
        "has_https": has_https, "has_ip": has_ip,
        "has_port": has_port, "has_www": has_www,
        "has_at_in_url": has_at_in_url, "subdomain_depth": subdomain_depth,
        "url_entropy": url_entropy, "hostname_entropy": hostname_entropy,
        "digit_ratio": digit_ratio, "special_char_ratio": special_char_ratio,
    }
    df = pd.DataFrame([features])
    df = df.fillna(0)
    return df


# --------------------------------------------------------------
# Model loading — identical pattern to predict.py
# --------------------------------------------------------------
def load_models():
    """
    Load all three trained classifiers and the StandardScaler
    from outputs/. Returns (lr, rf, dt, scaler) or raises
    FileNotFoundError with a clear message if any file is missing.
    """
    required = {                              # map display names to file paths for clear error messages
        "Logistic Regression": MODEL_LR,
        "Random Forest":       MODEL_RF,
        "Decision Tree":       MODEL_DT,
        "StandardScaler":      SCALER_PATH,
    }
    missing = [name for name, path in required.items()
               if not os.path.exists(path)]  # collect names of any missing artifact files

    if missing:                              # if any files are missing, raise an error with a clear message
        raise FileNotFoundError(
            f"Missing model files: {', '.join(missing)}\n\n"
            "Please run train_models.py first to generate these files."
        )

    with open(MODEL_LR,    "rb") as f: lr     = pickle.load(f)  # load Logistic Regression model
    with open(MODEL_RF,    "rb") as f: rf     = pickle.load(f)  # load Random Forest model
    with open(MODEL_DT,    "rb") as f: dt     = pickle.load(f)  # load Decision Tree model
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)  # load StandardScaler

    return lr, rf, dt, scaler


# --------------------------------------------------------------
# Main GUI Application class
# --------------------------------------------------------------
class PhishingDetectorApp:
    """
    The main GUI window for the phishing detection system.
    Built using tkinter — no additional packages required.
    """

    def __init__(self, root: tk.Tk):
        """
        Initialise the application window, load model artifacts,
        and build all GUI widgets.
        """
        self.root = root                     # store reference to the root tkinter window
        self.root.title("Phishing Detection System — URL Classifier")  # set the window title bar text
        self.root.configure(bg=BG_DARK)      # set the main window background to the dark theme colour
        self.root.resizable(True, True)      # allow the window to be resized by the user

        # Set a minimum window size to prevent widgets from being hidden
        self.root.minsize(750, 600)          # minimum width 750px, minimum height 600px

        # Try to set the window to open at a reasonable size
        self.root.geometry("900x700")        # initial window size: 900px wide, 700px tall

        # Load model artifacts at startup — show error and exit if missing
        try:
            self.lr, self.rf, self.dt, self.scaler = load_models()  # load all four model artifacts
            self.models_loaded = True        # flag that models loaded successfully for use in classify()
        except FileNotFoundError as e:
            self.models_loaded = False       # flag that models failed to load
            messagebox.showerror(            # show a popup error dialog with the missing files message
                "Model Files Not Found", str(e)
            )

        # Build all the GUI widgets
        self._build_header()                 # title bar at the top of the window
        self._build_input_section()          # URL input field and Classify button
        self._build_results_section()        # results panel showing per-model verdicts
        self._build_footer()                 # status bar at the bottom of the window

        # Set initial focus to the URL input field so user can type immediately
        self.url_entry.focus_set()

        # Bind the Enter key to trigger classification — more convenient than clicking
        self.root.bind("<Return>", lambda event: self._classify())


    def _build_header(self):
        """Build the title header at the top of the window."""
        header_frame = tk.Frame(             # create a frame to hold the header content
            self.root, bg=ACCENT_BLUE,       # blue background matching the dashboard theme
            pady=15                          # vertical padding inside the header frame
        )
        header_frame.pack(fill=tk.X)         # expand the header frame to fill the full window width

        tk.Label(                            # create the main title label
            header_frame,
            text="🛡  Phishing Detection System",  # shield emoji adds visual interest for presentation
            font=("Segoe UI", 18, "bold"),   # large bold font for the title
            bg=ACCENT_BLUE,                  # match the header background colour
            fg=TEXT_WHITE,                   # white text for contrast on the blue background
        ).pack()                             # place the label in the centre of the header frame

        tk.Label(                            # create the subtitle label below the main title
            header_frame,
            text="URL Classification using Machine Learning — 24 URL-String Features",
            font=("Segoe UI", 10),           # smaller regular font for the subtitle
            bg=ACCENT_BLUE,
            fg="#BDD7EE",                    # slightly muted blue-white for the subtitle text
        ).pack()                             # place below the main title


    def _build_input_section(self):
        """Build the URL input field and Classify button."""
        input_frame = tk.Frame(              # outer frame for the input section with padding
            self.root, bg=BG_DARK, pady=20, padx=30
        )
        input_frame.pack(fill=tk.X)          # expand to fill window width

        tk.Label(                            # label above the input field
            input_frame,
            text="Enter URL to classify:",
            font=("Segoe UI", 11, "bold"),
            bg=BG_DARK, fg=TEXT_WHITE,
        ).pack(anchor=tk.W)                  # anchor to the left (West) edge

        tk.Label(                            # smaller instructional text below the label
            input_frame,
            text="No network connection is made — the URL is analysed as a string only.",
            font=("Segoe UI", 9),
            bg=BG_DARK, fg=TEXT_GREY,
        ).pack(anchor=tk.W, pady=(0, 8))     # anchor left, small bottom padding

        # Frame to hold the input field and button side by side
        entry_row = tk.Frame(input_frame, bg=BG_DARK)
        entry_row.pack(fill=tk.X)            # expand to fill width

        # URL text entry field
        self.url_var = tk.StringVar()        # StringVar to hold and monitor the URL text
        self.url_entry = tk.Entry(           # create the text input field
            entry_row,
            textvariable=self.url_var,       # bind the entry to the StringVar
            font=("Consolas", 12),           # monospace font appropriate for URLs
            bg=BG_INPUT,                     # dark input background
            fg=TEXT_WHITE,                   # white text
            insertbackground=TEXT_WHITE,     # white cursor in the input field
            relief=tk.FLAT,                  # flat border style
            bd=8,                            # border/padding width inside the entry
        )
        self.url_entry.pack(                 # place the entry field, expanding to fill available width
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10)
        )

        # Classify button
        self.classify_btn = tk.Button(       # create the classify button
            entry_row,
            text="  Classify  ",             # button label with padding spaces for visual width
            font=("Segoe UI", 11, "bold"),
            bg=ACCENT_BLUE,                  # blue button matching the header
            fg=TEXT_WHITE,                   # white text
            activebackground="#1A5490",      # slightly darker blue when button is clicked
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,                  # flat style — modern look
            padx=15, pady=8,                 # internal padding for button size
            cursor="hand2",                  # change cursor to hand pointer on hover
            command=self._classify,          # call _classify() method when clicked
        )
        self.classify_btn.pack(side=tk.LEFT) # place the button to the right of the entry field

        # Clear button
        clear_btn = tk.Button(               # create the clear button to reset the input and results
            entry_row,
            text="  Clear  ",
            font=("Segoe UI", 11),
            bg=BG_INPUT,
            fg=TEXT_GREY,
            activebackground=BORDER,
            activeforeground=TEXT_WHITE,
            relief=tk.FLAT,
            padx=10, pady=8,
            cursor="hand2",
            command=self._clear,             # call _clear() method when clicked
        )
        clear_btn.pack(side=tk.LEFT, padx=(6, 0))


    def _build_results_section(self):
        """Build the results panel that displays classification verdicts."""
        # Outer container frame for the results section
        outer = tk.Frame(self.root, bg=BG_DARK, padx=30)
        outer.pack(fill=tk.BOTH, expand=True)  # expand to fill remaining window space

        tk.Label(                            # section heading label
            outer,
            text="Classification Results",
            font=("Segoe UI", 11, "bold"),
            bg=BG_DARK, fg=TEXT_WHITE,
        ).pack(anchor=tk.W, pady=(0, 8))

        # Results card — the panel that displays verdicts
        self.results_card = tk.Frame(        # the card frame with a dark panel background
            outer, bg=BG_PANEL,
            bd=1, relief=tk.FLAT,
        )
        self.results_card.pack(fill=tk.BOTH, expand=True)

        # ── Waiting state message (shown before first classification) ─
        self.waiting_frame = tk.Frame(self.results_card, bg=BG_PANEL)
        self.waiting_frame.pack(expand=True)  # centre in the card

        tk.Label(                            # placeholder text before any URL is classified
            self.waiting_frame,
            text="Enter a URL above and click Classify",
            font=("Segoe UI", 13),
            bg=BG_PANEL, fg=TEXT_GREY,
        ).pack(pady=40)

        # ── Results content frame (hidden until classification runs) ─
        self.results_frame = tk.Frame(self.results_card, bg=BG_PANEL)
        # Not packed yet — will be shown when results are available

        # URL display label inside results frame
        self.url_display_var = tk.StringVar()
        tk.Label(                            # shows the URL that was classified
            self.results_frame,
            textvariable=self.url_display_var,
            font=("Consolas", 10),
            bg=BG_PANEL, fg=TEXT_GREY,
            wraplength=800,                  # wrap long URLs to avoid horizontal overflow
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=20, pady=(15, 5))

        # Separator line
        tk.Frame(
            self.results_frame, bg=BORDER, height=1
        ).pack(fill=tk.X, padx=20, pady=5)

        # ── Per-model result rows ────────────────────────────────
        # Each model gets its own row with name, verdict, and confidence
        models_frame = tk.Frame(self.results_frame, bg=BG_PANEL)
        models_frame.pack(fill=tk.X, padx=20, pady=10)

        # Store verdict label variables for dynamic updates
        self.lr_verdict_var    = tk.StringVar()  # Logistic Regression verdict text
        self.rf_verdict_var    = tk.StringVar()  # Random Forest verdict text
        self.dt_verdict_var    = tk.StringVar()  # Decision Tree verdict text
        self.lr_conf_var       = tk.StringVar()  # LR confidence text
        self.rf_conf_var       = tk.StringVar()  # RF confidence text
        self.dt_conf_var       = tk.StringVar()  # DT confidence text
        self.lr_verdict_label  = None            # will hold the Label widget for colour changes
        self.rf_verdict_label  = None
        self.dt_verdict_label  = None

        # Build one row per model
        model_rows = [
            ("Logistic Regression",     self.lr_verdict_var, self.lr_conf_var, "lr"),
            ("Random Forest",           self.rf_verdict_var, self.rf_conf_var, "rf"),
            ("Decision Tree (depth=10)", self.dt_verdict_var, self.dt_conf_var, "dt"),
        ]

        for model_name, verdict_var, conf_var, key in model_rows:
            row = tk.Frame(models_frame, bg=BG_PANEL)   # frame for one model result row
            row.pack(fill=tk.X, pady=4)                  # stack rows vertically

            # Model name label — left aligned
            tk.Label(
                row,
                text=model_name,
                font=("Segoe UI", 11),
                bg=BG_PANEL, fg=TEXT_WHITE,
                width=26, anchor=tk.W,       # fixed width for alignment across all rows
            ).pack(side=tk.LEFT)

            # Verdict label — green for Legitimate, red for Malicious
            verdict_lbl = tk.Label(
                row,
                textvariable=verdict_var,
                font=("Segoe UI", 11, "bold"),
                bg=BG_PANEL, fg=TEXT_GREY,   # initially grey — updated when results come in
                width=12, anchor=tk.W,
            )
            verdict_lbl.pack(side=tk.LEFT)

            # Store reference to verdict label widget for colour updates
            if key == "lr":
                self.lr_verdict_label = verdict_lbl
            elif key == "rf":
                self.rf_verdict_label = verdict_lbl
            elif key == "dt":
                self.dt_verdict_label = verdict_lbl

            # Confidence score label — right of the verdict
            tk.Label(
                row,
                textvariable=conf_var,
                font=("Segoe UI", 10),
                bg=BG_PANEL, fg=TEXT_GREY,
            ).pack(side=tk.LEFT)

        # Separator before majority vote
        tk.Frame(
            self.results_frame, bg=BORDER, height=1
        ).pack(fill=tk.X, padx=20, pady=10)

        # ── Majority vote display ────────────────────────────────
        majority_row = tk.Frame(self.results_frame, bg=BG_PANEL)
        majority_row.pack(fill=tk.X, padx=20, pady=(0, 5))

        tk.Label(                            # "Majority Vote (2 of 3):" label
            majority_row,
            text="Majority Vote (2 of 3):",
            font=("Segoe UI", 13, "bold"),
            bg=BG_PANEL, fg=TEXT_WHITE,
            width=26, anchor=tk.W,
        ).pack(side=tk.LEFT)

        self.majority_var = tk.StringVar()   # StringVar for the majority vote verdict text
        self.majority_label = tk.Label(      # the majority verdict label — large and prominent
            majority_row,
            textvariable=self.majority_var,
            font=("Segoe UI", 14, "bold"),
            bg=BG_PANEL, fg=TEXT_GREY,       # initially grey — updated with colour when results come in
        )
        self.majority_label.pack(side=tk.LEFT)

        # ── Caution note (shown when classifiers disagree) ───────
        self.caution_var = tk.StringVar()
        self.caution_label = tk.Label(
            self.results_frame,
            textvariable=self.caution_var,
            font=("Segoe UI", 9, "italic"),
            bg=BG_PANEL, fg=YELLOW,          # yellow for caution messages
            wraplength=800,
            justify=tk.LEFT,
        )
        self.caution_label.pack(anchor=tk.W, padx=20, pady=(0, 15))


    def _build_footer(self):
        """Build the status bar at the bottom of the window."""
        footer = tk.Frame(                   # narrow frame at the very bottom
            self.root, bg=BG_PANEL,
            pady=6, bd=1, relief=tk.FLAT
        )
        footer.pack(fill=tk.X, side=tk.BOTTOM)

        # Status message — updated during classification
        self.status_var = tk.StringVar(value="Ready — enter a URL to classify.")
        tk.Label(
            footer,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            bg=BG_PANEL, fg=TEXT_GREY,
        ).pack(side=tk.LEFT, padx=15)

        # Model status indicator on the right side of the footer
        model_status = "✓ Models loaded" if self.models_loaded else "✗ Models not found — run train_models.py"
        model_colour = GREEN if self.models_loaded else RED
        tk.Label(
            footer,
            text=model_status,
            font=("Segoe UI", 9),
            bg=BG_PANEL, fg=model_colour,
        ).pack(side=tk.RIGHT, padx=15)


    def _classify(self):
        """
        Called when the Classify button is clicked or Enter is pressed.
        Extracts features from the URL, runs all three classifiers,
        and updates the results panel with verdicts and confidence scores.
        """
        if not self.models_loaded:           # if models failed to load, show error and stop
            messagebox.showerror(
                "Models Not Loaded",
                "Model files are missing. Please run train_models.py first."
            )
            return

        url = self.url_var.get().strip()     # get the URL text from the input field, stripping whitespace

        if not url:                          # if the input field is empty, show a warning and stop
            messagebox.showwarning("No URL", "Please enter a URL to classify.")
            return

        # Update status bar to show classification is in progress
        self.status_var.set("Classifying...")
        self.root.update_idletasks()         # force the GUI to refresh so the status update is visible immediately

        try:
            # Extract 24 URL-string features from the input URL
            features_df     = extract_features_single(url)

            # Apply the StandardScaler (transform only — no re-fitting) for Logistic Regression
            features_scaled = self.scaler.transform(features_df)

            # Run all three classifiers
            lr_pred  = self.lr.predict(features_scaled)[0]           # LR prediction (0 or 1)
            lr_proba = self.lr.predict_proba(features_scaled)[0]     # LR class probabilities

            rf_pred  = self.rf.predict(features_df)[0]               # RF prediction (unscaled)
            rf_proba = self.rf.predict_proba(features_df)[0]         # RF class probabilities

            dt_pred  = self.dt.predict(features_df)[0]               # DT prediction (unscaled)
            dt_proba = self.dt.predict_proba(features_df)[0]         # DT class probabilities

            # Compute majority vote
            votes    = lr_pred + rf_pred + dt_pred                   # sum of binary predictions (0-3)
            majority = "Malicious" if votes >= 2 else "Legitimate"   # majority verdict threshold

            # Update the GUI with the results
            self._update_results(
                url,
                lr_pred, lr_proba,
                rf_pred, rf_proba,
                dt_pred, dt_proba,
                majority, votes
            )

            self.status_var.set(f"Classification complete — {majority}")  # update status bar

        except Exception as e:               # catch any unexpected errors during classification
            messagebox.showerror(            # show error popup with the exception message
                "Classification Error",
                f"An error occurred during classification:\n{e}"
            )
            self.status_var.set("Error during classification.")


    def _update_results(self, url, lr_pred, lr_proba, rf_pred,
                        rf_proba, dt_pred, dt_proba, majority, votes):
        """
        Update all result labels with the new classification verdicts
        and switch from the waiting state to the results state.
        """
        # Switch from waiting placeholder to results content
        self.waiting_frame.pack_forget()         # hide the "Enter a URL" placeholder
        self.results_frame.pack(                 # show the results frame
            fill=tk.BOTH, expand=True
        )

        # Update the URL display
        self.url_display_var.set(f"URL: {url}")

        # Helper: convert prediction and probabilities to verdict + confidence
        def verdict_conf(pred, proba):
            verdict    = "Malicious" if pred == 1 else "Legitimate"
            confidence = proba[int(pred)] * 100
            return verdict, f"({confidence:.1f}% confidence)"

        lr_verdict, lr_conf = verdict_conf(lr_pred, lr_proba)
        rf_verdict, rf_conf = verdict_conf(rf_pred, rf_proba)
        dt_verdict, dt_conf = verdict_conf(dt_pred, dt_proba)

        # Update verdict text variables
        self.lr_verdict_var.set(lr_verdict)
        self.rf_verdict_var.set(rf_verdict)
        self.dt_verdict_var.set(dt_verdict)
        self.lr_conf_var.set(lr_conf)
        self.rf_conf_var.set(rf_conf)
        self.dt_conf_var.set(dt_conf)

        # Update verdict label colours (green = Legitimate, red = Malicious)
        self.lr_verdict_label.config(fg=GREEN if lr_verdict == "Legitimate" else RED)
        self.rf_verdict_label.config(fg=GREEN if rf_verdict == "Legitimate" else RED)
        self.dt_verdict_label.config(fg=GREEN if dt_verdict == "Legitimate" else RED)

        # Update majority vote display
        self.majority_var.set(majority)
        self.majority_label.config(          # colour the majority verdict prominently
            fg=GREEN if majority == "Legitimate" else RED
        )

        # Show or hide the caution note based on whether classifiers disagree
        if votes == 1 or votes == 2:         # classifiers disagree — 1 or 2 malicious votes out of 3
            self.caution_var.set(
                "⚠  Classifiers disagree on this URL. "
                "Treat the majority verdict with caution and consider manual review."
            )
        else:                                # all classifiers agree — clear the caution message
            self.caution_var.set("")


    def _clear(self):
        """
        Clear the URL input field and reset the results panel
        back to the waiting state.
        """
        self.url_var.set("")                 # clear the URL input field
        self.url_entry.focus_set()           # return focus to the input field for convenience

        # Switch back from results to waiting placeholder
        self.results_frame.pack_forget()     # hide the results frame
        self.waiting_frame.pack(expand=True) # show the "Enter a URL" placeholder again

        # Reset all result variables to empty strings
        for var in [self.lr_verdict_var, self.rf_verdict_var, self.dt_verdict_var,
                    self.lr_conf_var, self.rf_conf_var, self.dt_conf_var,
                    self.majority_var, self.caution_var, self.url_display_var]:
            var.set("")

        self.status_var.set("Ready — enter a URL to classify.")  # reset status bar


# --------------------------------------------------------------
# Entry point
# --------------------------------------------------------------
def main():
    """
    Create the tkinter root window and launch the application.
    Must be run from the project root directory so that the
    outputs/ path resolves correctly to find the .pkl files.
    """
    root = tk.Tk()                           # create the main tkinter window

    # Set the window icon if the icon file exists (optional — won't fail if missing)
    try:
        root.iconbitmap("icon.ico")          # set a custom window icon if one exists in the project root
    except Exception:
        pass                                 # silently skip if no icon file is found

    app = PhishingDetectorApp(root)          # create the application instance, which builds all widgets
    root.mainloop()                          # start the tkinter event loop — runs until window is closed


if __name__ == "__main__":
    main()                                   # run the GUI when the script is executed directly

