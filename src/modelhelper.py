import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
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
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = None
        
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
        df_cleaned = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        if len(df_cleaned) < initial_rows:
            self.logger.info(f"Removed {initial_rows - len(df_cleaned)} duplicate rows")
        
        # Handle missing values
        missing_counts = df_cleaned.isnull().sum()
        if missing_counts.any():
            self.logger.info(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
            
            # For numerical columns, fill with median
            numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numerical_cols] = df_cleaned[numerical_cols].fillna(df_cleaned[numerical_cols].median())
            
            # For categorical columns, fill with mode
            categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
            df_cleaned[categorical_cols] = df_cleaned[categorical_cols].fillna(df_cleaned[categorical_cols].mode().iloc[0])
        
        return df_cleaned

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
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col])
        
        # Scale numerical features if requested
        if scale_data and len(numerical_cols) > 0:
            # Fit and transform the scaler on numerical columns
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            
            # Ensure exact standardization
            for col in numerical_cols:
                X[col] = (X[col] - X[col].mean()) / X[col].std()
        
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
        
        # Ensure exact standardization for training data
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            X_train[col] = (X_train[col] - X_train[col].mean()) / X_train[col].std()
            # Apply the same scaling to test data
            X_test[col] = (X_test[col] - X_train[col].mean()) / X_train[col].std()
        
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   model_type: str = 'random_forest', task: str = 'classification') -> Any:
        """
        Train a specified model on the prepared data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            model_type (str): Type of model to train
            task (str): 'classification' or 'regression'
            
        Returns:
            Trained model instance
        """
        if task not in ['classification', 'regression']:
            raise ValueError("Task must be either 'classification' or 'regression'")
        
        if task == 'classification':
            models = {
                'random_forest': RandomForestClassifier(random_state=self.random_state),
                'logistic_regression': LogisticRegression(random_state=self.random_state),
                'svm': SVC(random_state=self.random_state)
            }
        else:
            models = {
                'random_forest': RandomForestRegressor(random_state=self.random_state),
                'linear_regression': LinearRegression(),
                'svr': SVR()
            }
        
        if model_type not in models:
            raise ValueError(f"Invalid model type. For {task}, choose from: {list(models.keys())}")
        
        model = models[model_type]
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
        if task not in ['classification', 'regression']:
            raise ValueError("Task must be either 'classification' or 'regression'")
        
        y_pred = model.predict(X_test)
        
        if task == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
        else:
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        
        return metrics

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
