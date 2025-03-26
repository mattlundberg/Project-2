import kagglehub
import kagglehub.auth
import pandas as pd
import os
import requests
from modelhelper import ModelHelper
from dotenv import load_dotenv
from typing import Tuple

class DataFetcherKAGGLE:
    """
    A class for fetching data from Kaggle datasets using kagglehub.
    """
    
    def __init__(self):
        self.hasFlightDataset = False
        self.flightDataset = None

    

    def clear_flight_data(self):
        """
        Clear flight data.
        """
        self.flightData = None
        
class DataFrameHelper:
    """
    A class for helping with pandas DataFrames.
    """

    def drop_columns(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Remove columns from a dataframe.
        """
        df = df.drop(columns=columns)

        return df