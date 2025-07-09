# EvaluationMetricsSD
Repository for paper on evaluation metrics for privacy-preserving synthetic data.

# Usage
- create a data/ directory
- download the following tables from MIMIC_IV v3.1 as gzipped csv and put them in the data/ directory: admissions, patients, omr, diagnoses_icd
- run utils/preprocess_mimic.py
- run main.py (alter number of cross validation folds, evaluation metrics, etc. as seen fit)