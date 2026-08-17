# IoT NIDS – Machine Learning Analysis

Machine learning experiments for anomaly detection in IoT network traffic using the **ToN-IoT dataset**.

This repository contains the data preprocessing, model training, evaluation, and visualization scripts developed as part of a Master's thesis focused on detecting anomalous network traffic in IoT environments.

## Overview

The project evaluates several supervised and unsupervised machine learning approaches for distinguishing normal and malicious IoT network traffic.

The following algorithms are included:

- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- Isolation Forest

The experiments include binary anomaly detection, feature analysis, class imbalance analysis, and evaluation using standard classification metrics.

## Dataset

The experiments are based on the **ToN-IoT Network dataset**, which contains normal and malicious network traffic collected from IoT/IIoT environments.

The preprocessing pipeline includes:

- duplicate removal
- handling of placeholder values
- removal of irrelevant and potentially leakage-prone features
- feature engineering
- one-hot encoding of categorical features
- stratified train/test split
- standardization of numerical features

Large CSV dataset files and generated train/test files are not included in this repository due to their size.

## Repository Structure

```text
.
├── scripts/
│   ├── clean_data.py
│   ├── preprocess_data.py
│   ├── train_supervised.py
│   ├── train_isolation_forest.py
│   ├── compare_class_weight.py
│   ├── feature_correlation.py
│   └── generate_plots.py
│
├── thesis_data/
│   └── generated data and results
│
├── requirements.txt
└── README.md
```

## Model Evaluation

The models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Additional experiments investigate class imbalance and the detection performance of minority attack categories, with particular attention to MITM (Man-in-the-Middle) attacks.

## Running the Experiments

### 1. Clone the repository

```bash
git clone <repository-url>
cd iot-nids-ml-analysis
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the required **ToN-IoT Network dataset** and place the source CSV file in the `thesis_data` directory.

Large dataset files are excluded from Git using `.gitignore` and therefore need to be downloaded separately.

### 5. Run data preprocessing

Run the data cleaning and preprocessing scripts first:

```bash
python scripts/clean_data.py
python scripts/preprocess_data.py
```

These scripts generate the processed train and test datasets required by the machine learning experiments.

### 6. Train and evaluate the models

Run the supervised models:

```bash
python scripts/train_supervised.py
```

Run the Isolation Forest experiment:

```bash
python scripts/train_isolation_forest.py
```

### 7. Run additional experiments

Class imbalance analysis:

```bash
python scripts/compare_class_weight.py
```

Feature correlation analysis:

```bash
python scripts/feature_correlation.py
```

### 8. Generate figures

The figures used for visual analysis can be generated with:

```bash
python scripts/generate_plots.py
```

Generated datasets, evaluation results, and figures are stored in the `thesis_data` directory.

## Web Application

An interactive web application was developed as a separate component of the project to demonstrate IoT network anomaly detection directly in the browser.

The web application allows users to explore network traffic data, train a machine learning model in the browser, visualize model performance, and analyze user-provided network traffic instances.

The web application is maintained in a separate repository:

**iot-anomaly-detection**

## Master's Thesis

This repository contains the machine learning and data analysis component of the Master's thesis:

**Development of an IoT Analytical Platform for Anomaly Detection in Smart Device Network Traffic**

The research focuses on the application and comparison of machine learning techniques for detecting anomalous network traffic in IoT environments.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Matplotlib

## Author

**Sandra Ristić**

## License

This project is available under the MIT License.