---

**Evaluation Report:**

**Accuracy:** 0.9434

- The model has an **overall accuracy** of 94.34%, meaning that 94.34% of the predictions were correct. This is a strong indicator of the model's overall performance.

---

**Classification Report:**

|              | Precision | Recall  | F1-Score | Support |
|--------------|-----------|---------|----------|---------|
| **Class 0**  | 0.92      | 0.97    | 0.94     | 56463   |
| **Class 1**  | 0.97      | 0.91    | 0.94     | 56839   |
| **Accuracy** |           |         | 0.94     | 113302  |
| **Macro avg**| 0.95      | 0.94    | 0.94     | 113302  |
| **Weighted avg** | 0.95  | 0.94    | 0.94     | 113302  |

---

### **Interpretation:**

#### **Precision:**
- **Class 0 (No fraud)**: The **precision** for Class 0 is 0.92, meaning that 92% of the predicted non-fraudulent transactions were correct. The model performs well in predicting normal transactions.
- **Class 1 (Fraud)**: The **precision** for Class 1 is 0.97, indicating that when the model predicts a transaction as fraudulent, it is correct 97% of the time. This shows the model is highly effective at identifying fraud when it occurs.

#### **Recall:**
- **Class 0 (No fraud)**: The **recall** for Class 0 is 0.97, meaning that the model identified 97% of all real non-fraudulent transactions. This indicates that very few non-fraudulent transactions were missed.
- **Class 1 (Fraud)**: The **recall** for Class 1 is 0.91, meaning that the model correctly identified 91% of all fraudulent transactions. Although this is strong, there is room for improvement, as some fraudulent transactions were not detected (false negatives).

#### **F1-Score:**
- **Class 0 (No fraud)**: The **F1-score** for Class 0 is 0.94, which reflects a good balance between precision and recall for non-fraudulent transactions.
- **Class 1 (Fraud)**: The **F1-score** for Class 1 is 0.94, indicating a good balance between precision and recall for fraudulent transactions. An F1-score near 1.0 suggests the model is finding a good trade-off between detecting fraud and avoiding false positives.

#### **Average Metrics:**
- **Macro Average**: The **macro average** gives a global measure of performance by averaging the scores for each class without considering class size. Here, the **macro F1-score** is 0.94, indicating that the model performs well across both classes.
- **Weighted Average**: The **weighted average** takes into account the proportion of each class in the dataset. The **weighted F1-score** is also 0.94, suggesting that the model is well-balanced in terms of performance across both classes, even though Class 0 (non-fraud) is likely more frequent.

---

**Confusion Matrix:**

```
[[55008  1455]
 [ 4955 51884]]
```

#### **Confusion Matrix Interpretation:**
- **True Negatives (TN)**: 55,008 correctly predicted non-fraudulent transactions.
- **False Positives (FP)**: 1,455 incorrectly predicted fraudulent transactions for non-fraudulent instances.
- **False Negatives (FN)**: 4,955 fraudulent transactions that were not detected by the model.
- **True Positives (TP)**: 51,884 correctly predicted fraudulent transactions.

### **Conclusion:**

The model performs well with **high precision** and **recall** for both classes. It is particularly effective at identifying fraudulent transactions with a balanced **F1-score** and good **precision** and **recall** for both classes. However, there is still some room for improvement in identifying fraudulent transactions, as evidenced by the false negatives. The confusion matrix shows that the model minimizes false positives, which is important to avoid flagging legitimate transactions as fraudulent. Overall, the model appears to be well-suited for fraud detection with strong performance across the board.

---