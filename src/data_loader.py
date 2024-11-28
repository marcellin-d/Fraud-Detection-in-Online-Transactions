import pandas as pd

def load_data(file_path, **kwargs):
    """
    Load dataset from a specified file path.
    
    Parameters:
    ----------
    file_path : str
        Path to the CSV file.
    **kwargs : dict
        Additional keyword arguments for pandas.read_csv.
    
    Returns:
    -------
    pd.DataFrame
        Loaded dataset as a Pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path, **kwargs)
        print(f"Data successfully loaded from {file_path}. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: No data found in file at {file_path}.")
        raise
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        raise
