from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

def train_model(X_train, y_train, model_type='logistic', **kwargs):
    """
    Train a model with the specified type.
    
    Parameters:
    ----------
    X_train : pd.DataFrame or np.ndarray
        The training feature set.
    y_train : pd.Series or np.ndarray
        The training target labels.
    model_type : str, optional
        The type of model to train ('logistic' or 'random_forest').
        Default is 'logistic'.
    **kwargs : dict
        Additional keyword arguments to pass to the model constructor.
    
    Returns:
    -------
    model : sklearn.base.BaseEstimator
        The trained model.
    
    Raises:
    ------
    ValueError
        If an unsupported model type is provided.
    """
    try:
        if model_type == 'logistic':
            model = LogisticRegression(**kwargs)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        model.fit(X_train, y_train)
        print(f"Model of type '{model_type}' trained successfully.")
        return model
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        raise
    except NotFittedError as nfe:
        print(f"Model fitting failed: {nfe}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during model training: {e}")
        raise
