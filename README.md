---

# **Credit Card Fraud Detection**

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The goal is to predict whether a given transaction is legitimate or fraudulent based on various features of the transaction. The dataset used in this project includes anonymized features for privacy, such as the transaction amount, time, and other factors that may contribute to identifying fraudulent activities.

---

## **Table of Contents**

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Dependencies](#dependencies)
- [Running the Project](#running-the-project)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)
- [Handling Imbalanced Dataset with SMOTE](#handling-imbalanced-dataset-with-smote)
- [Contact](#contact)

---

## **Project Overview**

In this project, machine learning models are applied to a real-world dataset of credit card transactions to detect fraud. The entire process follows a typical data science pipeline:

1. **Data Loading and Exploration**
2. **Data Preprocessing** (Handling missing values, scaling, etc.)
3. **Handling Imbalanced Dataset** (SMOTE)
4. **Model Training** (Logistic Regression, Random Forest)
5. **Evaluation** (Accuracy, Precision, Recall, Confusion Matrix)

At the end of the project, we obtain a trained model, performance evaluation metrics, and a detailed report summarizing the results.

---

## **Project Structure**

```plaintext
├── data/                      # Data files
│   ├── raw/                   # Raw data files
│   └── processed/             # Processed data files
├── notebooks/                 # Jupyter notebooks for exploratory analysis
├── src/                       # Source code for the project
│   ├── data_loader.py         # Functions for loading the data
│   ├── preprocess.py          # Functions for data preprocessing
│   ├── model.py               # Functions for training models
│   ├── evaluate.py            # Functions for evaluating the model
│   ├── utils.py               # Utility functions for data handling
├── evaluation_report.txt      # Evaluation results and interpretation
├── requirements.txt           # List of dependencies
├── main.py                    # Main script to execute the project
└── README.md                  # Project overview and documentation
```

---

## **Setup and Installation**

To get started with the project, follow the steps below:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/CreditCardFraudDetection.git
cd CreditCardFraudDetection
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

- On **Windows**:
  ```bash
  venv\Scripts\activate
  ```

- On **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## **Dependencies**

The following Python libraries are required to run this project:

- **pandas**
- **numpy**
- **scikit-learn**
- **matplotlib**
- **seaborn**
- **imblearn** (for SMOTE)

To install all dependencies at once:

```bash
pip install -r requirements.txt
```

---

## **Running the Project**

To run the project and generate results, execute the `main.py` script. This script will handle the entire pipeline from data loading to model evaluation.

```bash
python main.py
```

The script will output the following:

1. Data loading confirmation, including the shape of the dataset.
2. Data preprocessing steps, such as handling missing values and scaling the "Amount" column.
3. Model training results for Logistic Regression and Random Forest classifiers.
4. Evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

---

## **Model Evaluation**

After running the model, review the detailed evaluation in the `evaluation_report.txt` file. The evaluation includes:

### Key Metrics:

- **Accuracy**: 94.34%  
- **Precision** for both classes (fraud and non-fraud)  
- **Recall** for both classes  
- **F1-Score** for a balanced measure of precision and recall  
- **Confusion Matrix**: Provides insights into true positives, false positives, true negatives, and false negatives.

### Example Output:

```
Accuracy: 0.9434

Classification Report:
              precision    recall  f1-score   support

           0       0.92      0.97      0.94     56463
           1       0.97      0.91      0.94     56839

    accuracy                           0.94    113302
   macro avg       0.95      0.94      0.94    113302
weighted avg       0.95      0.94      0.94    113302

Confusion Matrix:
[[55008  1455]
 [ 4955 51884]]
```

---

## **Conclusion**

This project demonstrates the use of machine learning to tackle the problem of credit card fraud detection. By utilizing models like **Logistic Regression** and **Random Forest**, we can identify fraudulent transactions with an impressive accuracy of 94.34%. The provided evaluation metrics give a detailed view of how the model performs, helping improve fraud detection systems in real-world scenarios.

---

## **Handling Imbalanced Dataset with SMOTE**

One of the main challenges encountered in this project was the **imbalanced dataset**. The dataset contains far more non-fraudulent transactions (Class 0) than fraudulent transactions (Class 1). This imbalance can lead to biased models that predict the majority class more frequently, undermining the detection of fraud.

To address this issue, we employed **SMOTE** (Synthetic Minority Over-sampling Technique) from the `imblearn` library. SMOTE generates synthetic samples of the minority class (fraudulent transactions) by interpolating between existing examples, thereby balancing the dataset and improving the model's ability to correctly identify fraudulent transactions.

### SMOTE Code Example

Here is a code snippet demonstrating how SMOTE is applied to balance the dataset:

```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load your dataset
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Now X_res and y_res are balanced and ready for model training
```

This technique helps ensure the model doesn't become biased toward predicting the majority class (non-fraudulent transactions), ultimately improving fraud detection performance.

---

## 📫 Contact

For questions or suggestions, feel free to reach out:  
- **Name**: Marcellin DJAMBO
- **Email**: djambomarcellin@gmail.com
- **LinkedIn**: [My LinkedIn Profile](https://www.linkedin.com/in/marcellindjambo)

---
