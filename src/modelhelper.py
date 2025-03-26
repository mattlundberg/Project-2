import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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
        print(categorical_cols)
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        print(numerical_cols)
        
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
                'logistic_regression': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,  # Increased from default 100
                    class_weight='balanced',  # Handle class imbalance
                    solver='lbfgs',  # Use L-BFGS solver
                    multi_class='multinomial',  # Handle multiple classes
                    penalty='l2',  # Use L2 regularization
                    C=0.1 # Regularization strength
                ),
                'svm': SVC(random_state=self.random_state),
                'decision_tree': DecisionTreeClassifier(
                    random_state=self.random_state,
                    max_depth=10,  # Limit tree depth to prevent overfitting
                    min_samples_split=5,  # Minimum samples required to split
                    min_samples_leaf=2,  # Minimum samples in leaf nodes
                    class_weight='balanced',  # Handle class imbalance
                    criterion='gini'  # Use Gini impurity for classification
                )
            }
        else:
            models = {
                'random_forest': RandomForestRegressor(random_state=self.random_state),
                'linear_regression': LinearRegression(),
                'svr': SVR(),
                'decision_tree': DecisionTreeRegressor(random_state=self.random_state)
            }
        
        if model_type not in models:
            raise ValueError(f"Invalid model type. For {task}, choose from: {list(models.keys())}")
        
        model = models[model_type]
        
        # Ensure data is scaled for logistic regression
        if model_type == 'logistic_regression':
            X_train_scaled = self.scaler.fit_transform(X_train)
            model.fit(X_train_scaled, y_train)
        else:
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

    def predict(self, model: Any, airline: str, departure_date: str, origin: str) -> str:
        """
        Make predictions for a specific flight using the trained model.
        
        Args:
            model: Trained model instance
            airline (str): Airline name
            departure_date (str): Departure date in 'YYYY-MM-DD' format
            origin (str): Origin airport code
            
        Returns:
            str: Predicted delay category ('Early', 'On Time', or 'Delayed')
        """
        # Create a DataFrame with the input features in the correct order
        input_data = pd.DataFrame({
            'AIRLINE': [airline],
            'ORIGIN': [origin],
            'DEP_DELAY': [0],
            'DAY_OF_WEEK': [pd.to_datetime(departure_date).dayofweek],
            'TOTAL_DELAY': [0],
        })
        
        # Ensure categorical columns are properly encoded
        for col in ['AIRLINE', 'ORIGIN']:
            if col in self.label_encoders:
                input_data[col] = self.label_encoders[col].transform(input_data[col])
        
        # Get the feature names from the model's training data
        feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else input_data.columns
        
        # Reorder columns to match training data
        input_data = input_data[feature_names]
        
        # Scale numerical features
        numerical_cols = input_data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            input_data[numerical_cols] = self.scaler.transform(input_data[numerical_cols])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Convert numeric prediction to category
        if hasattr(self.label_encoders, 'DELAY_CATEGORY'):
            prediction = self.label_encoders['DELAY_CATEGORY'].inverse_transform([prediction])[0]
        
        return prediction

    def optimize_flight_dataset_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize data types for the flight dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with optimized data types
        """
        # Get the top 5 airlines by frequency
        top_airlines = ['Southwest Airlines Co.', 'American Airlines Inc.', 'Delta Air Lines Inc.', 'Spirit Air Lines', 'Allegiant Air']
        df = df[df['AIRLINE'].isin(top_airlines)]
        
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
    
    def prepare_flight_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the flight dataset for modeling.
        """
        print("Optimizing flight dataset data types...")
        df = self.optimize_flight_dataset_dtypes(df)

        # Drop columns not needed for modeling
        columns_to_drop = [
            'AIRLINE_DOT',    # Redundant airline identifier
            'AIRLINE_CODE',   # Redundant airline identifier  
            'DOT_CODE',       # Redundant airline identifier
            'FL_NUMBER',      # Flight number not predictive
            'CRS_DEP_TIME',   # Will extract hour instead
            'TAXI_OUT',       # Not available at prediction time
            'WHEELS_OFF',     # Not available at prediction time  
            'WHEELS_ON',      # Not available at prediction time
            'TAXI_IN',        # Not available at prediction time
            'CRS_ARR_TIME',   # Not needed for departure delay prediction
            'ARR_TIME',       # Not available at prediction time
            'ARR_DELAY',      # Not available at prediction time
            'CRS_ELAPSED_TIME', # Will use distance instead
            'AIR_TIME',       # Not available at prediction time
            'DISTANCE',       # Keeping for route complexity calculation
            'DEST',          # Using DEST_CITY instead
            'DEST_CITY',     # Destination may affect delays
            'ELAPSED_TIME',  # Not available at prediction time
            'DEP_TIME',      # Not available at prediction time
            'ORIGIN_CITY',   # Origin may affect delays
            'CANCELLED',     # Not predicting cancellations
            'CANCELLATION_CODE', # Not predicting cancellations
            'DIVERTED'       # Not predicting diversions
        ]
        df = df.drop(columns=columns_to_drop)
        
        # Feature Engineering
        print("Performing feature engineering...")
        
        # Create time-based features from scheduled departure time
        df['DAY_OF_WEEK'] = pd.to_datetime(df['FL_DATE']).dt.dayofweek
        
        # Create delay categories based on departure delay
        df['DELAY_CATEGORY'] = pd.cut(df['DEP_DELAY'], 
                                    bins=[-np.inf, -15, 15, np.inf],
                                    labels=['Early', 'On Time', 'Delayed'])
        
        # Create total delay feature from all delay types
        delay_columns = ['DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 
                        'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 
                        'DELAY_DUE_LATE_AIRCRAFT']
        df['TOTAL_DELAY'] = df[delay_columns].sum(axis=1)
        # Handle class imbalance for delay categories
        print("Handling class imbalance...")
        delay_counts = df['DELAY_CATEGORY'].value_counts()
        
        # Check if we have any data
        if len(delay_counts) == 0:
            raise ValueError("No data available after preprocessing")
            
        max_delay = delay_counts.max()
        
        # Balance the dataset using oversampling
        balanced_dfs = []
        for category in df['DELAY_CATEGORY'].unique():
            category_df = df[df['DELAY_CATEGORY'] == category]
            if len(category_df) > 0:  # Only process non-empty categories
                if len(category_df) < max_delay:
                    # Oversample minority classes
                    category_df = category_df.sample(n=max_delay, replace=True, random_state=self.random_state)
                balanced_dfs.append(category_df)
        
        if not balanced_dfs:
            raise ValueError("No valid data categories found after balancing")
            
        df = pd.concat(balanced_dfs)
        
        # Drop remaining columns not needed for modeling
        final_columns_to_drop = ['FL_DATE'] + delay_columns
        df = df.drop(columns=final_columns_to_drop, errors='ignore')
        
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