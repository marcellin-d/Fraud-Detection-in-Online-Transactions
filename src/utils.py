import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE

def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split data into training and testing sets.
    
    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing features and target.
    target_column : str
        The name of the target column to predict.
    test_size : float, optional
        The proportion of the dataset to include in the test split. Default is 0.2.
    random_state : int, optional
        The random seed for reproducibility. Default is 42.
    
    Returns:
    -------
    tuple
        Four objects: X_train, X_test, y_train, y_test.
    
    Raises:
    ------
    KeyError
        If the target column is not found in the DataFrame.
    ValueError
        If the DataFrame is empty.
    """
    if df.empty:
        raise ValueError("The input DataFrame is empty.")
    
    if target_column not in df.columns:
        raise KeyError(f"The target column '{target_column}' is not found in the DataFrame.")
    
    # Splitting the data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_res, y_res = SMOTE().fit_resample(X,y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=test_size, random_state=random_state)
    
    print(f"Data split completed: Train set - {X_train.shape[0]} samples, Test set - {X_test.shape[0]} samples.")
    return X_train, X_test, y_train, y_test
