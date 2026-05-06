# Privacy-Preserving Offline Machine Learning System for Malicious URL Detection

## Overview
This repository contains the implementation of a privacy-preserving, fully offline machine learning system for malicious URL detection. The system classifies URLs as malicious or legitimate using only features derived from the URL string itself, without relying on external data sources such as DNS queries, WHOIS lookups, webpage content, or API-based threat intelligence.

The design emphasizes reproducibility, interpretability, and deployment in constrained environments, including air-gapped systems and privacy-sensitive contexts. The system is implemented as a modular pipeline composed of independent scripts, each responsible for a single stage of processing.

---

## Key Characteristics

- **Fully Offline Operation**
  - No network communication at any stage
  - No DNS resolution, WHOIS queries, or HTTP requests

- **Privacy-Preserving Design**
  - No external data enrichment or data sharing
  - All processing occurs locally

- **Reproducible Pipeline**
  - Deterministic execution using fixed random seeds
  - Intermediate outputs stored as CSV files for inspection and validation

- **Modular Architecture**
  - Eight independent scripts following the single-responsibility principle
  - File-based data flow between pipeline stages

- **Lightweight Machine Learning Models**
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - Majority-vote ensemble for inference

---

## System Architecture

The system is implemented as an eight-stage pipeline:

1. **Data Ingestion**
   - `convert_phishtank.py` — Parses PhishTank XML into CSV format
   - `convert_urlhaus.py` — Parses URLhaus text feed into CSV format

2. **Dataset Construction**
   - `build_dataset.py` — Combines malicious and legitimate URLs and enforces class balance

3. **Data Preparation**
   - `prepare_data.py` — Cleans, validates, and deduplicates datasets

4. **Feature Engineering**
   - `extract_features.py` — Extracts 24 lexical and structural features from URL strings

5. **Model Training**
   - `train_models.py` — Trains and evaluates classifiers using stratified sampling and cross-validation

6. **Cross-Dataset Evaluation**
   - `cross_dataset_eval.py` — Evaluates generalization across distinct threat intelligence sources

7. **Inference**
   - `predict.py` — Classifies new URLs using trained models
   - `phishing_gui.py` — Optional graphical interface for offline classification

All scripts operate independently and communicate exclusively through CSV files and serialized model artifacts.

---

## Feature Engineering

The system derives **24 features** exclusively from the URL string, including:

- Length-based features (e.g., URL length, path length)
- Character count features (e.g., number of dots, digits)
- Structural indicators (e.g., presence of HTTPS, IP address usage)
- Statistical measures (e.g., entropy, character ratios)

A reduced feature set of 13 features may also be used based on feature importance analysis.

---

## Machine Learning Models

Three lightweight classifiers are implemented:

- **Logistic Regression**
  - Linear baseline model with feature scaling

- **Random Forest**
  - Ensemble model capturing nonlinear feature interactions

- **Decision Tree**
  - Interpretable rule-based classifier

For inference, the system uses a **majority-vote ensemble**, combining predictions from all three models.

---

## Data Sources

This system is designed to operate on publicly available datasets:

- PhishTank — Verified phishing URLs  
- URLhaus — Malware distribution URLs  
- Tranco Top Sites — Legitimate domains  

Due to size and licensing considerations, these datasets are not included in the repository.

Users must download the datasets from their official sources and place them in the `data/` directory.

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/crodriguez1221/privacy-preserving-url-detection.git
cd privacy-preserving-url-detection
```
2. Create a virtual environment:
```bash
python -m venv venv
```
3. Activate the environment:
```bash
venv\Scripts\activate
```
4. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### Run the pipeline

```bash
python src/prepare_data.py
python src/extract_features.py
python src/train_models.py
python src/cross_dataset_eval.py
```

### Predict a single URL

```bash
python src/predict.py "http://example.com/login"
```

### Optional GUI

```bash
python src/phishing_gui.py
```
---

## Repository Structure

```text
.
├── src/              # Pipeline scripts
├── data/             # Input datasets (not included)
├── outputs/          # Model artifacts and results (not included)
├── notebooks/        # Optional exploratory work
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Reproducibility 

- All random processes use `random_state = 42`
- Feature extraction is deterministic
- Preprocessing is applied consistently across datasets
- Results can be reproduced by following the pipeline with identical inputs

---

## Limitations

- The system relies exclusively on URL-string-derived features and does not incorporate:
  - Webpage content analysis
  - DNS or WHOIS metadata
  - External threat intelligence feeds
- Cross-dataset evaluation shows that models trained on one dataset may not fully generalize to structurally different URL distributions

---

## Intended Use

This repository is intended for:

- Academic research and demonstration
- Exploration of lightweight, offline malicious URL detection
- Study of cross-dataset generalization in cybersecurity

---

## Author

Connie Rodriguez  
Master of Science in Computer Science  
Saint Martin’s University