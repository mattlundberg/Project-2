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
    
class DataFetcherNOAA:
    """
    A class for fetching data from NOAA datasets using NOAA API.
    """
    
    def __init__(self):
        self.noaaData = None
        self.noaa_api_key = 'bjJHSvSZxSmURbGZiokSEwfqFjPTpsUQ'

    def fetch_noaa_data(self) -> pd.DataFrame:
        """
        Fetch NOAA data from NOAA API.
        
        Returns:
            pd.DataFrame: Weather data from NOAA
        """
        if self.noaaData is None:
            print("Fetching NOAA data from NOAA API...")
            response = requests.get(
                f"https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&startdate=2021-06-01&enddate=2023-12-31&limit=1000",
                headers={"token": self.noaa_api_key}
            )
            
            # Check if request was successful
            response.raise_for_status()
            
            # Convert JSON response to DataFrame directly from the results list
            self.noaaData = pd.DataFrame(response.json()['results'])
            
            # Convert date column to datetime
            if 'date' in self.noaaData.columns:
                self.noaaData['date'] = pd.to_datetime(self.noaaData['date'])

        return self.noaaData

    def clear_noaa_data(self):
        """
        Clear NOAA data.
        """
        self.noaaData = None

