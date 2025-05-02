import pandas as pd
import os


def load_kc_house_data(csv_path=None):
    """
    Loads the kc_house_data.csv file into a pandas DataFrame.

    Args:
        csv_path (str, optional): Path to the kc_house_data.csv file.
                                  If None, looks in the current directory.

    Returns:
        pd.DataFrame: DataFrame containing the house data.
    """
    if csv_path is None:
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "kc_house_data.csv"
        )
        csv_path = os.path.abspath(csv_path)
    df = pd.read_csv(csv_path)
    return df
