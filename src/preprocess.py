import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, output_path=None):
    """
    Preprocess the data, including scaling and handling missing values.
    
    Parameters:
    ----------
    df : pd.DataFrame
        The input dataset to preprocess.
    output_path : str, optional
        Path to save the processed DataFrame as a CSV file. Default is None.
    
    Returns:
    -------
    pd.DataFrame
        The preprocessed DataFrame.
    """
    # Handle duplicates
    df = df.drop_duplicates()
    
    # Drop unnecessary columns
    if 'Time' in df.columns:
        df = df.drop("Time", axis=1)
    
    # Scale the 'Amount' column
    scaler = StandardScaler()
    if 'Amount' in df.columns:
        df["Amount"] = scaler.fit_transform(df[["Amount"]])
    else:
        raise KeyError("Column 'Amount' not found in the DataFrame.")
    
    #Oversampling
    
    
    # Save the processed data if an output path is provided
    if output_path:
        try:
            df.to_csv(output_path, index=False)
            print(f"Processed data successfully saved to {output_path}.")
        except Exception as e:
            print(f"Failed to save processed data: {e}")
            raise
    
    return df
