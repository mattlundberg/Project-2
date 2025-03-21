import kagglehub
import kagglehub.auth
import pandas as pd
import os
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

    def fetch_flight_dataset(self) -> pd.DataFrame:
        """
        Fetch flight dataset from Kaggle using kagglehub.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        df = self.flightDataset

        if not self.hasFlightDataset:
            print("Fetching flight dataset from Kaggle...")
            df = self.fetch_dataset(os.environ['KAGGLE_FLIGHT_FILE_PATH'], os.environ['KAGGLE_FLIGHT_FILE_NAME'])
            self.hasFlightDataset = True

        self.flightDataset = df

        if self.flightDataset is None:
            raise ValueError("Flight dataset not found.")
        
        print("Preparing flight dataset...")
        modelHelper = ModelHelper()
        df_optmized = modelHelper.prepare_flight_dataset(df)

        
        return df_optmized

    def fetch_dataset(self, file_path: str, file_name: str) -> pd.DataFrame:
        """
        Fetch dataset from Kaggle using kagglehub.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        # Load environment variables
        load_dotenv()
        
        # Set Kaggle credentials
        os.environ['KAGGLE_USERNAME'] = 'matthewlundberg'
        os.environ['KAGGLE_KEY'] = 'a879198a80d0e04e423e703539a74851'
        
        # Download latest version
        path = kagglehub.dataset_download(
            file_path
        )
        path = path + '/' + file_name

        # Load dataset
        return pd.read_csv(path, engine='python')