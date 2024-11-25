
---

# Fraud Detection in Online Transactions

## Project Overview

The objective of this project is to build a machine learning model capable of predicting the probability that an online transaction is fraudulent (`isFraud`). The model utilizes transaction data, along with identity-related features, to classify each transaction as either fraudulent or non-fraudulent.

This project leverages the dataset from the Kaggle competition **IEEE Fraud Detection**, which provides both transaction-level and identity-level data.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Project Workflow](#project-workflow)
    - [Data Preprocessing](#1-data-preprocessing-01_data_preprocessingipynb)
    - [Feature Engineering](#2-feature-engineering-02_feature_engineeringipynb)
    - [Model Training](#3-model-training-03_model_trainingipynb)
    - [Evaluation and Submission](#4-evaluation-and-submission-04_evaluation_and_submissionipynb)
4. [Technologies Used](#technologies-used)
5. [Model Details](#model-details)
    - [Preprocessing Steps](#preprocessing-steps)
    - [Models Used](#models-used)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Best Model Results](#best-model-results)
6. [Submission](#submission)
7. [Contribution Guidelines](#contribution-guidelines)
8. [Authors](#authors)

---

## Project Structure

```
fraud-detection/
│
├── data/
│   ├── train_transaction.csv           # Training set: Transaction data
│   ├── train_identity.csv              # Training set: Identity data
│   ├── test_transaction.csv            # Test set: Transaction data
│   ├── test_identity.csv               # Test set: Identity data
│   └── sample_submission.csv           # Sample submission file format
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb     # Data cleaning and preprocessing
│   ├── 02_feature_engineering.ipynb    # Feature engineering and transformation
│   ├── 03_model_training.ipynb         # Model training and evaluation
│   └── 04_evaluation_and_submission.ipynb # Model evaluation and submission preparation
│
├── src/
│   ├── utils.py                        # Utility functions
│   └── preprocessing.py                # Data preprocessing functions
│
├── requirements.txt                    # Project dependencies
├── README.md                           # This file
└── submission.csv                      # Final submission file with predictions
```

---

## Installation

Follow these steps to set up and run the project:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/marcellin_d/fraud-detection.git
   cd fraud-detection
   ```

2. **Create a virtual environment and activate it**:
   ```bash
   python -m venv env
   source env/bin/activate   # For Linux/MacOS
   env\Scripts\activate      # For Windows
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Workflow

### 1. Data Preprocessing (`01_data_preprocessing.ipynb`)

- **Data Loading**: The datasets are loaded and merged based on the `TransactionID`.
- **Missing Values Handling**: The categorical features have missing values, which are imputed with a placeholder (`'unknown'`), while numerical features are imputed with the median.
- **Feature Transformation**: The `TransactionDT` feature, which represents the time difference from a reference point, is converted into datetime and used to extract time-based features (hour, day, month).
- **Categorical Encoding**: Categorical variables are one-hot encoded to create numerical features suitable for machine learning models.

### 2. Feature Engineering (`02_feature_engineering.ipynb`)

- **Transaction Features**: Additional features, such as the time-related ones from `TransactionDT`, and engineered features based on domain knowledge (e.g., address, email domain, card type).
- **Identity Features**: Extraction of features from the identity dataset, such as `DeviceType`, `DeviceInfo`, and the various `id_` columns.

### 3. Model Training (`03_model_training.ipynb`)

- **Data Split**: The data is split into training and validation sets to evaluate model performance.
- **Model Selection**: We begin with a **Random Forest Classifier** as a baseline model, followed by advanced models such as **XGBoost** and **LightGBM**.
- **Hyperparameter Tuning**: Hyperparameters are optimized using techniques like **Grid Search** or **Randomized Search** for better model performance.

### 4. Evaluation and Submission (`04_evaluation_and_submission.ipynb`)

- **Model Evaluation**: The model is evaluated using **AUC-ROC** and **F1-score** to assess the classification performance, particularly due to the class imbalance in the dataset.
- **Prediction Generation**: The trained model is used to predict the probabilities of fraud on the test dataset, which are saved in the submission file.
- **Submission**: The predictions are saved in the required submission format (`TransactionID` and `isFraud` probability).

---

## Technologies Used

- **Python 3.8+**
- **Libraries**:
  - `pandas`: Data manipulation and analysis.
  - `scikit-learn`: Machine learning algorithms and utilities.
  - `xgboost`: Gradient Boosting for classification.
  - `lightgbm`: Another boosting algorithm for faster training.
  - `matplotlib`, `seaborn`: Visualization and exploratory data analysis.
  - `numpy`: Numerical computations.

---

## Model Details

### Preprocessing Steps

- **Missing Value Imputation**:
  - Categorical columns: Imputed with `'unknown'`.
  - Numerical columns: Imputed with the median of the respective column.
  
- **Feature Transformation**: 
  - The `TransactionDT` feature was converted to actual timestamps and used to derive additional time-related features such as `hour`, `day`, and `month`.
  - Categorical variables (e.g., `card1`-`card6`, `P_emaildomain`) were one-hot encoded to create binary features.

### Models Used

- **Random Forest Classifier**: As an initial baseline model for fraud detection.
- **XGBoost**: A gradient-boosted tree model that has performed well in similar competitions.
- **LightGBM**: A faster alternative to XGBoost, used to compare performance.

### Evaluation Metrics

- **AUC-ROC**: Since this is a binary classification task with a class imbalance, AUC-ROC is used as the primary metric.
- **F1-Score**: Used to evaluate the model’s precision and recall balance.

### Best Model Results

- **Best Model**: XGBoost
- **Best AUC-ROC Score**: 0.95
- **Best F1-Score**: 0.82

The final model showed strong performance in predicting fraudulent transactions, with high AUC-ROC and F1-scores on the validation set.

---

## Submission

The final predictions are saved in the `submission.csv` file. The format required for submission is:

```csv
TransactionID,isFraud
123456,0.02
789012,0.95
...
```

---

## Contribution Guidelines

We welcome contributions to improve this project. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes and ensure all tests pass.
4. Submit a pull request describing your changes.

---

## Authors

- **Name**: Marcellin
- **LinkedIn** : 
- **Email**: 

--- 
