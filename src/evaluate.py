from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, output_path=None):
    """
    Evaluate the model and return performance metrics.
    
    Parameters:
    ----------
    model : sklearn.base.BaseEstimator
        The trained model to evaluate.
    X_test : pd.DataFrame or np.ndarray
        The testing feature set.
    y_test : pd.Series or np.ndarray
        The true labels for the test set.
    output_path : str, optional
        Path to save the evaluation report. Default is None.
    
    Returns:
    -------
    dict
        A dictionary containing accuracy, classification report, and confusion matrix.
    
    Raises:
    ------
    NotFittedError
        If the model is not fitted before evaluation.
    """
    try:
        # Generate predictions
        predictions = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_test, predictions)
        
        # Compile evaluation results
        evaluation_results = {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist()
        }
        
        # Optional: Save evaluation report as a file
        if output_path:
            with open(output_path, 'w') as f:
                f.write("Accuracy: {:.4f}\n".format(acc))
                f.write("\nClassification Report:\n")
                f.write(classification_report(y_test, predictions))
                f.write("\nConfusion Matrix:\n")
                f.write(str(conf_matrix))
            print(f"Evaluation report saved to {output_path}.")
        
        print(f"Model evaluation completed. Accuracy: {acc:.4f}")
        return evaluation_results
    
    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")
        raise
