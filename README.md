# Privacy-Preserving Offline Machine Learning System for Malicious URL Detection

## Overview
This repository contains the core implementation of a privacy-preserving, fully offline machine learning system for malicious URL detection. The system classifies URLs as malicious or legitimate using only features derived from the URL string itself, without relying on external data sources such as DNS queries, WHOIS lookups, webpage content, or API-based threat intelligence.

The design emphasizes reproducibility, interpretability, and deployment in constrained environments, including air-gapped systems and privacy-sensitive contexts. The system is implemented as a modular pipeline composed of independent scripts, each responsible for a single stage of processing.

This project was developed as an Advanced Design Project for the Master of Science in Computer Science program at Saint Martin's University. 

---

## Intended Use

This repository demonstrates the design and evaluation of a privacy-preserving machine learning pipeline for cybersecurity research, with emphasis on:

- Malicious URL detection
- Offline and privacy-preserving system design
- Machine learning model development and evaluation
- Feature engineering and feature selection
- Cross-dataset generalization
- Reproducible experimental methodology
- Modular Python software design

This project is an academic research prototype and is not intended to replace production threat-detection or enterprise security systems.

---

## Key Characteristics

- **Fully Offline Operation**
  - No network communication during pipeline execution or inference
  - No DNS resolution, WHOIS queries, HTTP requests, or external API calls
  - All classification features are derived directly from the URL string

- **Privacy-Preserving Design**
  - No external data enrichment or runtime data sharing
  - All processing occurs locally after source datasets have been obtained

- **Reproducible Pipeline**
  - Fixed random seeds are used for randomly determined operations
  - Deterministic feature extraction
  - Intermediate outputs are stored for inspection and validation

- **Modular Architecture**
  - Eight core Python scripts following the single-responsibility principle
  - File-based communication between pipeline stages
  - Individual stages can be inspected, validated, or rerun independently

- **Lightweight Machine Learning Models**
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - Majority-vote ensemble for inference

- **Cross-Dataset Evaluation**
  - Bidirectional evaluation between independently sourced malicious URL datasets
  - Designed to measure generalization beyond the distribution used for training
    
---

## System Architecture

The core system consists of eight Python scripts organized into four logical processing phases followed by a standalone inference path.

### Phase 1 — Data Ingestion

- `convert_phishtank.py` — Parses PhishTank XML and converts verified malicious URLs into a standardized CSV format
- `convert_urlhaus.py` — Parses the URLhaus text feed and converts malicious URLs into the same standardized schema

### Phase 2 — Dataset Construction

- `build_dataset.py` — Independently combines each malicious source with legitimate domains from the Tranco Top Sites list and constructs balanced datasets

### Phase 3 — Data Preparation and Feature Engineering

- `prepare_data.py` — Cleans, validates, deduplicates, and normalizes dataset labels
- `extract_features.py` — Extracts 24 lexical and structural features from each URL string

### Phase 4 — Modeling and Evaluation

- `train_models.py` — Trains Logistic Regression, Random Forest, and Decision Tree classifiers using an 80/20 stratified train/test split and five-fold cross-validation
- `cross_dataset_eval.py` — Performs bidirectional cross-dataset evaluation to measure generalization across distinct malicious URL sources

### Standalone Inference

- `predict.py` — Loads trained model artifacts and classifies user-supplied URLs using the same URL-string feature engineering applied during training

The repository also includes a lightweight Tkinter graphical interface (`phishing_gui.py`) for demonstrating offline URL classification. The GUI uses the same underlying inference logic as `predict.py` and is provided as an optional interface rather than a component of the eight-script core pipeline.

Pipeline stages communicate through structured CSV files and serialized model artifacts rather than direct inter-script function calls, shared memory, or database connections.

---

## Feature Engineering

The baseline system derives **24 lexical and structural features** exclusively from the URL string.

These include:

- Length-based features
- Character-count features
- Structural URL indicators
- Entropy measurements
- Character-ratio measurements

Because all features can be calculated from the URL string itself, classification does not require resolving, visiting, or externally enriching the URL.

### Feature Selection Experiment

The project also evaluated a reduced feature configuration.

The experimental methodology consisted of three iterations:

1. **Iteration 1 — Baseline**
   - Full 24-feature schema
   - Baseline configurations for all three classifiers

2. **Iteration 2 — Feature Selection**
   - Random Forest feature importance was analyzed across both datasets
   - 11 low-contribution features were removed
   - The resulting **13-feature configuration** was evaluated using the same models

3. **Iteration 3 — Hyperparameter Optimization**
   - Grid-search hyperparameter optimization was applied using the reduced 13-feature configuration
   - Models were reevaluated to determine whether tuning improved cross-dataset generalization

The 13-feature configuration therefore represents an experimentally derived reduced feature set rather than the original feature-extraction schema.

---

## Machine Learning Models

Three lightweight supervised classifiers are implemented:

### Logistic Regression

A linear classifier used with feature scaling. `StandardScaler` is fitted on the training data and applied to evaluation data without fitting on the test set.

### Random Forest

An ensemble classifier capable of modeling nonlinear relationships between URL features. Random Forest feature importance was also used during the feature-selection experiment.

### Decision Tree

An interpretable rule-based classifier that provides a comparatively transparent decision structure.

### Majority-Vote Inference

For standalone inference, predictions from Logistic Regression, Random Forest, and Decision Tree are combined using a **majority-vote ensemble**.

Individual model predictions remain available alongside the combined classification result.

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

## Data Setup

### One-Time Dataset Preparation

The following scripts are used to convert the raw threat datasets and construct the experimental datasets:

```bash
python src/convert_phishtank.py
python src/convert_urlhaus.py
python src/build_dataset.py
```

These scripts generally need to be run only once for a given set of source data. They should be rerun if the source datasets are replaced or the experimental datasets need to be rebuilt.

---

## Usage

### Run the Modeling Pipeline

After the experimental datasets have been constructed:

```bash
python src/prepare_data.py
python src/extract_features.py
python src/train_models.py
python src/cross_dataset_eval.py

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

## Documentation

The `docs/` directory contains selected excerpts from the academic and technical documentation developed for this project, including:

- Software Requirements Specification (SRS) excerpt
- Final project report excerpt
- Presentation slides

The complete SRS and final report were produced as part of the Master of Science in Computer Science Advanced Design Project but are not included in this public repository.

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

## Author

Connie Rodriguez  
Master of Science in Computer Science  
Saint Martin’s University
