import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import logging
from typing import Union, Tuple, Optional, Dict, Any
import warnings

class ModelHelper:
    """
    A helper class for preprocessing datasets and preparing them for machine learning models.
    Includes data cleaning, feature engineering, and model selection capabilities.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ModelHelper with available models and preprocessing tools.
        
        Args:
            random_state (int): Seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = None
        self.label_encoders = {}
        self.imputer = None
        
        # Available models for classification and regression
        self.classification_models = {
            'random_forest': RandomForestClassifier(random_state=random_state),
            'logistic_regression': LogisticRegression(random_state=random_state),
            'svm': SVC(random_state=random_state)
        }
        
        self.regression_models = {
            'random_forest': RandomForestRegressor(random_state=random_state),
            'linear_regression': LinearRegression(),
            'svr': SVR()
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def clean_data(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values, removing duplicates,
        and identifying outliers.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Name of the target variable
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Create a copy to avoid modifying the original dataframe
        df_clean = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        if len(df_clean) < initial_rows:
            self.logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # Handle missing values
        missing_stats = df_clean.isnull().sum()
        if missing_stats.any():
            self.logger.info("Handling missing values...")
            self.imputer = SimpleImputer(strategy='mean')
            numeric_columns = df_clean.select_dtypes(include=['float64', 'int64']).columns
            df_clean[numeric_columns] = self.imputer.fit_transform(df_clean[numeric_columns])
        
        # Remove constant columns
        constant_columns = [col for col in df_clean.columns if df_clean[col].nunique() == 1]
        if constant_columns:
            df_clean = df_clean.drop(columns=constant_columns)
            self.logger.info(f"Removed constant columns: {constant_columns}")
        
        return df_clean

    def preprocess_features(self, df: pd.DataFrame, target_column: str, 
                          scale_data: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess features by encoding categorical variables and scaling numerical features.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Name of the target variable
            scale_data (bool): Whether to scale numerical features
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Preprocessed features and target variable
        """
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col])
        
        # Scale numerical features
        if scale_data:
            self.scaler = StandardScaler()
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        return X, y

    def prepare_data(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2,
                    scale_data: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for modeling by cleaning, preprocessing, and splitting into train/test sets.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Name of the target variable
            test_size (float): Proportion of dataset to include in the test split
            scale_data (bool): Whether to scale numerical features
            
        Returns:
            Tuple containing train/test splits for features and target
        """
        # Clean the data
        df_cleaned = self.clean_data(df, target_column)
        
        # Preprocess features
        X, y = self.preprocess_features(df_cleaned, target_column, scale_data)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   model_type: str = 'random_forest', task: str = 'classification',
                   **model_params) -> Any:
        """
        Train a specified model on the prepared data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            model_type (str): Type of model to train
            task (str): 'classification' or 'regression'
            **model_params: Additional parameters for the model
            
        Returns:
            Trained model instance
        """
        models = self.classification_models if task == 'classification' else self.regression_models
        
        if model_type not in models:
            raise ValueError(f"Model type '{model_type}' not supported. "
                          f"Available models: {list(models.keys())}")
        
        model = models[model_type]
        if model_params:
            model.set_params(**model_params)
        
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                      task: str = 'classification') -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        Args:
            model: Trained model instance
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            task (str): 'classification' or 'regression'
            
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        if task == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            return {'accuracy': accuracy}
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return {'mse': mse, 'r2': r2}

    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            model: Trained model instance
            X (pd.DataFrame): Features to predict
            
        Returns:
            np.ndarray: Predictions
        """
        return model.predict(X)
