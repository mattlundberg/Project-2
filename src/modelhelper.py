# Standard library imports
import os
import logging
import pickle
import warnings
from typing import Union, Tuple, Optional, Dict, Any, List

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import kagglehub.auth
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_RANDOM_STATE = 42
CUTOFF_DATE = '2021-06-15'
DELAY_BINS = [-np.inf, -15, 15, np.inf]
DELAY_LABELS = ['Early', 'On Time', 'Delayed']

class ModelHelper:
    """
    A helper class for preprocessing datasets and preparing them for machine learning models.
    Includes data cleaning, feature engineering, and model selection capabilities.
    """
    
    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE):
        """
        Initialize ModelHelper with preprocessing tools and model configurations.
        
        Args:
            random_state (int): Seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputer = None
        self.flight_dataset = None
        self.original_flight_dataset = None
        self.has_flight_dataset = False
        self.logger = logger
        self.model = None
        self.airlinedelay = None
        self.airportdelay = None
        self.xdata = None
        self.ydata = None

    def _get_model_config(self, model_type: str, task: str) -> Dict[str, Any]:
        """Get model configuration based on type and task."""
        if task not in ['classification', 'regression']:
            raise ValueError("Task must be either 'classification' or 'regression'")
        
        if task == 'classification':
            return {
                'random_forest': RandomForestClassifier(
                    random_state=self.random_state,
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    max_features='sqrt',
                    class_weight='balanced',
                    bootstrap=True,
                    oob_score=True
                ),
                'logistic_regression': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=2000,
                    class_weight='balanced',
                    solver='saga',
                    multi_class='multinomial',
                    penalty='l1',
                    C=1.0,
                    tol=1e-4
                ),
                'svm': SVC(random_state=self.random_state),
                'decision_tree': DecisionTreeClassifier(
                    random_state=self.random_state,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    criterion='gini'
                )
            }
        else:
            return {
                'random_forest': RandomForestRegressor(random_state=self.random_state),
                'linear_regression': LinearRegression(),
                'svr': SVR(),
                'decision_tree': DecisionTreeRegressor(random_state=self.random_state)
            }

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df_cleaned = df.copy()
        missing_counts = df_cleaned.isnull().sum()
        
        if missing_counts.any():
            self.logger.info(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
            
            # Handle numerical columns
            numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numerical_cols] = df_cleaned[numerical_cols].fillna(
                df_cleaned[numerical_cols].median()
            )
            
            # Handle categorical columns
            categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
            df_cleaned[categorical_cols] = df_cleaned[categorical_cols].fillna(
                df_cleaned[categorical_cols].mode().iloc[0]
            )
        
        return df_cleaned

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder."""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])
        return df

    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            for col in numerical_cols:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        return df

    def _balance_dataset(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Balance the dataset using oversampling."""
        delay_counts = df[target_column].value_counts()
        if len(delay_counts) == 0:
            raise ValueError("No data available after preprocessing")
            
        max_delay = delay_counts.max()
        balanced_dfs = []
        
        for category in df[target_column].unique():
            category_df = df[df[target_column] == category]
            if len(category_df) > 0:
                if len(category_df) < max_delay:
                    category_df = category_df.sample(
                        n=max_delay, replace=True, random_state=self.random_state
                    )
                balanced_dfs.append(category_df)
        
        if not balanced_dfs:
            raise ValueError("No valid data categories found after balancing")
            
        return pd.concat(balanced_dfs)

    def _prepare_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Prepare features for modeling."""
        # Create time-based features
        df['DAY_OF_WEEK'] = pd.to_datetime(df['FL_DATE']).dt.dayofweek
        
        # Create delay categories
        df['DELAY_CATEGORY'] = pd.cut(
            df['DEP_DELAY'], 
            bins=DELAY_BINS,
            labels=DELAY_LABELS
        )
        
        # Create total delay feature
        delay_columns = [
            'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 
            'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 
            'DELAY_DUE_LATE_AIRCRAFT'
        ]
        df['TOTAL_DELAY'] = df[delay_columns].sum(axis=1)

        # Drop delay columns
        df.drop(columns=delay_columns, inplace=True)
        
        return df

    def _drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns not needed for modeling."""
        columns_to_drop = [
            'AIRLINE_DOT', 'AIRLINE_CODE', 'DOT_CODE', 'FL_NUMBER',
            'CRS_DEP_TIME', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON',
            'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY',
            'CRS_ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'DEST',
            'DEST_CITY', 'ELAPSED_TIME', 'DEP_TIME', 'ORIGIN_CITY',
            'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED', 'FL_DATE',
            'DEP_DELAY'
        ]
        return df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    def _calculate_delay_statistics(self, df: pd.DataFrame) -> None:
        """
        Calculate and store mean delay statistics by airline and origin airport.
        
        Args:
            df (pd.DataFrame): Input dataframe containing flight data
        """
        # Calculate mean delay by airline
        self.airlinedelay = df.groupby('AIRLINE')['DEP_DELAY'].mean().to_dict()
        self.logger.info(f"Calculated mean delays for {len(self.airlinedelay)} airlines")
        
        # Calculate mean delay by origin airport
        self.airportdelay = df.groupby('ORIGIN')['DEP_DELAY'].mean().to_dict()
        self.logger.info(f"Calculated mean delays for {len(self.airportdelay)} airports")

    def prepare_flight_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the flight dataset for modeling."""

        self.logger.info("Optimizing flight dataset data types...")
        df = self._optimize_flight_dataset_dtypes(df)

        self.logger.info("Calculating delay statistics...")
        self._calculate_delay_statistics(df)
        
        self.logger.info("Performing feature engineering...")
        df = self._prepare_features(df, 'DELAY_CATEGORY')
        
        self.logger.info("Handling class imbalance...")
        df = self._balance_dataset(df, 'DELAY_CATEGORY')
        
        df = self._drop_unnecessary_columns(df)
        self.flight_dataset = df
        
        return df

    def train_model(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: pd.Series,
                   model_type: str = 'random_forest', task: str = 'classification') -> Any:
        """Train a specified model on the prepared data."""
        models = self._get_model_config(model_type, task)
        
        if model_type not in models:
            raise ValueError(f"Invalid model type. For {task}, choose from: {list(models.keys())}")
        
        model = models[model_type]
        
        self.logger.info(f"Training {model_type} model for {task} task")
        
        # Log features only if X_train is a DataFrame
        if isinstance(X_train, pd.DataFrame):
            self.logger.info(f"Features: {X_train.columns.tolist()}")
        
        if y_train is not None:
            self.logger.info(f"Target: {y_train.name}")
        
        model.fit(X_train, y_train)
        return model

    def predict(self, airline: str, departure_date: str, origin: str) -> str:
        """
        Make predictions for a specific flight.
        
        Args:
            airline (str): Airline name
            departure_date (str): Date of departure
            origin (str): Origin airport code
            
        Returns:
            str: Predicted delay category ('Early', 'On Time', or 'Delayed')
        """
        if self.model is None:
            raise ValueError("No model has been trained or loaded. Please train or load a model first.")
        # Get airline and airport delay statistics
        airline_delay = self.airlinedelay.get(airline, 0)
        airport_delay = self.airportdelay.get(origin, 0)
        
        # Calculate average delay
        total_delay = np.mean([airline_delay, airport_delay])
        self.logger.info(f"Airline delay: {airline} {airline_delay}, Airport delay: {origin} {airport_delay}, Total delay: {total_delay}")
        
        # Create input data
        input_data = pd.DataFrame({
            'AIRLINE': [airline],
            'ORIGIN': [origin],
            'DAY_OF_WEEK': [pd.to_datetime(departure_date).dayofweek],
            'TOTAL_DELAY': [total_delay]
        })
        
        # Encode categorical features
        for col in ['AIRLINE', 'ORIGIN']:
            if col in self.label_encoders:
                try:
                    input_data[col] = self.label_encoders[col].transform(input_data[col])
                except ValueError:
                    self.logger.warning(f"Unknown {col}: {input_data[col].iloc[0]}")
                    input_data[col] = 0
        
        # Scale numerical features only for logistic regression
        if isinstance(self.model, LogisticRegression):
            self.logger.info("Applying scaling for logistic regression prediction")
            numerical_cols = input_data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                input_data[numerical_cols] = self.scaler.transform(input_data[numerical_cols])
        
        # Make prediction
        prediction = self.model.predict(input_data)[0]
        
        # Convert numeric prediction to category
        if 'DELAY_CATEGORY' in self.label_encoders:
            prediction = self.label_encoders['DELAY_CATEGORY'].inverse_transform([prediction])[0]
        
        return prediction

    def _optimize_flight_dataset_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize data types for the flight dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with optimized data types
        """
        # Get the top 5 airlines by frequency
        #top_airlines = ['Southwest Airlines Co.', 'American Airlines Inc.', 'Delta Air Lines Inc.', 'Spirit Air Lines', 'Allegiant Air']
        #df = df[df['AIRLINE'].isin(top_airlines)]
        
        # Remove records before June 15, 2021
        cutoff_date = pd.to_datetime('2021-06-15')
        df = df[pd.to_datetime(df['FL_DATE']) >= cutoff_date]

        
        # Convert date columns
        date_columns = ['FL_DATE']
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert time columns
        time_columns = ['CRS_DEP_TIME', 'DEP_TIME', 'WHEELS_OFF', 'WHEELS_ON', 
                       'CRS_ARR_TIME', 'ARR_TIME']
        for col in time_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert delay columns to float
        delay_columns = ['DEP_DELAY', 'ARR_DELAY', 'DELAY_DUE_CARRIER', 
                        'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS', 
                        'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT']
        for col in delay_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert boolean columns
        boolean_columns = ['CANCELLED', 'DIVERTED']
        for col in boolean_columns:
            df[col] = df[col].astype(bool)
        
        # Convert categorical columns
        categorical_columns = ['AIRLINE', 'AIRLINE_DOT', 'AIRLINE_CODE', 'DOT_CODE',
                             'ORIGIN', 'ORIGIN_CITY', 'DEST', 'DEST_CITY', 'CANCELLATION_CODE']
        for col in categorical_columns:
            df[col] = df[col].astype('category')
        
        # Convert numeric columns
        numeric_columns = ['FL_NUMBER', 'TAXI_OUT', 'TAXI_IN', 'CRS_ELAPSED_TIME',
                          'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        delay_columns = ['DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT']
        df[delay_columns] = df[delay_columns].fillna(0)
        
        return df
    
    def save_model(self, model: Any, model_name: str) -> None:
        """
        Save a trained model to the models directory.
        
        Args:
            model (Any): Trained model object to save
            model_name (str): Name to give the saved model file
        """
        import os
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Add .pkl extension if not provided
        if not model_name.endswith('.pkl'):
            model_name += '.pkl'
            
        model_path = os.path.join('models', model_name)
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            self.logger.info(f"Model successfully saved to {model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model to {model_path}: {str(e)}")
            raise

    def load_model(self, model_name: str) -> Any:
        """
        Load a saved model from the models directory.
        
        Args:
            model_name (str): Name of the saved model file
            
        Returns:
            Any: Loaded model object
        """ 
        import os

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)    

        # Add .pkl extension if not provided
        if not model_name.endswith('.pkl'):
            model_name += '.pkl'

        model_path = os.path.join('models', model_name) 

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.logger.info(f"Model successfully loaded from {model_path}")
            return model        
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise  

    def fetch_flight_dataset(self) -> pd.DataFrame:
        """
        Fetch flight dataset from Kaggle using kagglehub.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """

        if self.flight_dataset is not None:
            self.logger.info("Flight dataset is already loaded and optimized")
            return self.flight_dataset
        
        df = self.flight_dataset

        if not self.has_flight_dataset:
            print("Fetching flight dataset from Kaggle...")
            df = self.fetch_dataset(os.environ['KAGGLE_FLIGHT_FILE_PATH'], os.environ['KAGGLE_FLIGHT_FILE_NAME'])
            print("Preparing flight dataset...")
            self.original_flight_dataset = df.copy()
            df_optmized = self.prepare_flight_dataset(df)
            self.has_flight_dataset = True

        self.flight_dataset = df

        if self.flight_dataset is None:
            raise ValueError("Flight dataset not found.")
        
        return df_optmized

    def fetch_dataset(self, file_path: str, file_name: str) -> pd.DataFrame:
        """
        Fetch dataset from Kaggle using kagglehub.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        # Load environment variables
        load_dotenv()
        
        # Download latest version
        path = kagglehub.dataset_download(
            file_path
        )
        path = path + '/' + file_name

        # Load dataset
        return pd.read_csv(path, engine='python')    

    def train_flight_delay_model(self, model_type: str = 'logistic_regression', test_size: float = 0.2) -> Tuple[Any, Dict[str, float]]:
        """
        Orchestrate the complete process of retrieving data and training a flight delay prediction model.
        """
        self.logger.info("Starting flight delay model training process...")
        
        if self.flight_dataset is None:
            # Step 1: Fetch and prepare the dataset
            self.logger.info("Step 1: Fetching and preparing the dataset...")
            df = self.fetch_flight_dataset()
            self.logger.info(f"Dataset shape: {df.shape}")
            
            # Step 2: Prepare features and target
            self.logger.info("Step 2: Preparing features and target...")
            X = df.drop(columns=['DELAY_CATEGORY'])
            y = df['DELAY_CATEGORY']

            #Save the data for later use and to avoid issues when fetching a already loaded dataset.
            self.xdata = X.copy()   
            self.ydata = y.copy() 
        else:
            self.logger.info("Flight dataset is already loaded. Using the loaded dataset...")
            df = self.flight_dataset 
        
        # Log original features for reference
        self.logger.info(f"Features being used: {self.xdata.columns.tolist()}")
        
        # Step 3: Split the data
        self.logger.info(f"Step 3: Splitting data into train/test sets (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.xdata, self.ydata, test_size=test_size, random_state=self.random_state
        )
        self.logger.info(f"Training set shape: {X_train.shape}")
        self.logger.info(f"Test set shape: {X_test.shape}")
        
        # Step 4: Preprocess the data
        self.logger.info("Step 4: Preprocessing features...")
        # First encode categorical features
        X_train_processed = self._encode_categorical_features(X_train.copy())
        X_test_processed = self._encode_categorical_features(X_test.copy())
        
        # Then scale if using logistic regression
        if model_type == 'logistic_regression':
            X_train_processed = self.scaler.fit_transform(X_train_processed)
            X_test_processed = self.scaler.transform(X_test_processed)
        
        # Step 5: Train the model
        self.logger.info(f"Step 5: Training {model_type} model...")
        model = self.train_model(X_train_processed, y_train, model_type=model_type)
        self.model = model
        
        # Step 6: Evaluate the model
        self.logger.info("Step 6: Evaluating model performance...")
        y_pred = model.predict(X_test_processed)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Log the results
        self.logger.info("\nModel Evaluation Results:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric.capitalize()}: {value:.4f}")
        
        # Step 7: Save the model
        self.logger.info(f"Step 7: Saving the trained model as '{model_type}_flight_delay_model.pkl'...")
        self.save_model(model, f"{model_type}_flight_delay_model")
        
        self.logger.info("\nModel training process completed successfully!")
        return model, metrics    