import pandas as pd
import os

def load_data(file_path:str)->pd.DataFrame:
    """
    Loads CSV data into a pandas Dataframe
    
    Args:
        file_path(str) - path of the csv file
    
    Returns:
        pd.Dataframe: Loaded Dataset
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path)