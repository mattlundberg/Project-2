# Standard library imports
import os
import logging
import pickle
from typing import Union, Tuple, Dict, Any

# Third-party imports
import pandas as pd
import numpy as np
import kagglehub
import kagglehub.auth
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_RANDOM_STATE = 32
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
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=50,
                    min_samples_leaf=20,
                    max_features='sqrt',
                    class_weight='balanced_subsample',
                    bootstrap=True,
                    oob_score=True,
                    n_jobs=-1
                ),
                'logistic_regression': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced',
                    solver='lbfgs',
                    multi_class='multinomial',
                    C=1.0,
                    tol=1e-3,
                    n_jobs=-1,
                    warm_start=True
                ),
                'svm': SVC(
                    random_state=self.random_state,
                    kernel='rbf',
                    class_weight='balanced',
                    probability=True,
                    cache_size=1000
                ),
                'decision_tree': DecisionTreeClassifier(
                    random_state=self.random_state,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
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
        """
        Prepare features for modeling by creating time-based features and delay categories.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing flight data
            target_column (str): Name of the target column for prediction
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        # Extract time-based features
        df['DAY_OF_YEAR'] = pd.to_datetime(df['FL_DATE']).dt.dayofyear
        df['DAY_OF_WEEK'] = pd.to_datetime(df['FL_DATE']).dt.dayofweek
        
        # Create delay features
        delay_columns = [
            'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 
            'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 
            'DELAY_DUE_LATE_AIRCRAFT'
        ]
        
        # Calculate various delay statistics
        df['TOTAL_DELAY'] = df[delay_columns].sum(axis=1)
        #df['MAX_DELAY'] = df[delay_columns].max(axis=1)
        df['DELAY_VARIANCE'] = df[delay_columns].var(axis=1)
        
        # Calculate historical delay patterns
        df['HISTORICAL_DELAY'] = df.groupby(['AIRLINE', 'ORIGIN', 'DAY_OF_WEEK'])['TOTAL_DELAY'].transform('mean')
        
        # Define the target variable (DELAY_CATEGORY) based on DEP_DELAY
        df['DELAY_CATEGORY'] = pd.cut(
            df['DEP_DELAY'],
            bins=DELAY_BINS,
            labels=DELAY_LABELS
        )
        
        # Drop original delay columns
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
            'DEP_DELAY', 'DAY_OF_WEEK'
        ]
        return df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    
    def prepare_flight_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the flight dataset for modeling."""

        self.logger.info("Optimizing flight dataset data types...")
        df = self._optimize_flight_dataset_dtypes(df)
        
        self.logger.info("Performing feature engineering...")
        df = self._prepare_features(df, 'DELAY_CATEGORY')
        
        self.logger.info("Handling class imbalance...")
        df = self._balance_dataset(df, 'DELAY_CATEGORY')
        
        # After preparing the dataset
        class_distribution = df['DELAY_CATEGORY'].value_counts()
        self.logger.info(f"Class distribution before balancing:\n{class_distribution}")

        # After balancing
        if hasattr(self.model, 'classes_'):
            self.logger.info("Model classes:", self.model.classes_)
        
        df = self._drop_unnecessary_columns(df)
        self.flight_dataset = df
        
        return df

    def train_model(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: pd.Series,
                   model_type: str = 'random_forest', task: str = 'classification', force_train: bool = False) -> Any:
        """Train a specified model on the prepared data."""

        #Check to see if there is a model already trained and loaded.
        self.model = self.load_model(f"{model_type}_flight_delay_model")
        if self.model is not None and not force_train:
            self.logger.info(f"Model {model_type} already trained and loaded. Using the loaded model...")
            return self.model

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

    def predict(self, airline: str, departure_date: str, origin: str) -> Dict[str, Any]:
        """
        Make predictions for a specific flight.
        
        Args:
            airline (str): Airline name
            departure_date (str): Date of departure in yyyy-mm-dd format
            origin (str): Origin airport code
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - prediction: Predicted delay category ('Early', 'On Time', or 'Delayed')
                - probabilities: Dictionary of probabilities for each class
                - features: Dictionary of input features used for prediction
        """
        if self.model is None:
            raise ValueError("No model has been trained or loaded. Please train or load a model first.")
        
        # Convert departure date to day of year
        day_of_year = pd.to_datetime(departure_date).dayofyear

        # Calculate total delay
        total_delay = self._calculate_total_delay_by_day(day_of_year, airline, origin)
        
        # Log to see the values
        self.logger.info(f"Airline: {airline}, Airport: {origin}, Total delay: {total_delay}")

        # Create input data with all required features
        input_data = pd.DataFrame({
            'AIRLINE': [airline],
            'ORIGIN': [origin], 
            'DAY_OF_YEAR': [day_of_year],
            'TOTAL_DELAY': [total_delay],
            'DELAY_VARIANCE': [-20.0],  # Set to 0 for single prediction
            'HISTORICAL_DELAY': [self._get_historical_delay(airline, origin)]
        })
        
        # Ensure feature order matches training data
        if hasattr(self.model, 'feature_names_in_'):
            input_data = input_data[self.model.feature_names_in_]
        
        # Encode categorical features
        for col in ['AIRLINE', 'ORIGIN']:
            if col in self.label_encoders:
                try:
                    input_data[col] = self.label_encoders[col].transform(input_data[col])
                except ValueError:
                    self.logger.warning(f"Unknown {col}: {input_data[col].iloc[0]}")
                    input_data[col] = 0
        
        # Handle NaN values
        input_data = input_data.fillna(0)
        
        # Scale numerical features only for logistic regression
        if isinstance(self.model, LogisticRegression):
            self.logger.info("Applying scaling for logistic regression prediction")
            numerical_cols = input_data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                input_data[numerical_cols] = self.scaler.transform(input_data[numerical_cols])
        
        # Make prediction
        prediction = self.model.predict(input_data)[0]
        
        # Get prediction probabilities if available
        probabilities = {}
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(input_data)[0]
            probabilities = dict(zip(self.model.classes_, proba))
        
        # Convert numeric prediction to category
        if 'DELAY_CATEGORY' in self.label_encoders:
            prediction = self.label_encoders['DELAY_CATEGORY'].inverse_transform([prediction])[0]
        
        return {
            'prediction': prediction,
            'probabilities': ', '.join(f'{k}: {v*100:.2f}%' for k,v in probabilities.items())
        }

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
            
        Raises:
            Exception: If there is an error saving the model
        """
        import os
        
        # Ensure model name has .pkl extension
        model_name = f"{model_name}.pkl" if not model_name.endswith('.pkl') else model_name
        
        # Construct full path to save model
        model_path = os.path.join('models', model_name)
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        try:
            # Save model using pickle serialization
            with open(model_path, 'wb') as model_file:
                pickle.dump(model, model_file)
            self.logger.info(f"Model successfully saved to {model_path}")
        except Exception as e:
            # Log and re-raise any errors that occur during saving
            self.logger.error(f"Error saving model to {model_path}: {str(e)}")
            raise

    def load_model(self, model_name: str) -> Any:
        """
        Load a saved model from the models directory.
        
        Args:
            model_name (str): Name of the saved model file
            
        Returns:
            Any: Loaded model object
            
        Raises:
            FileNotFoundError: If model file cannot be found
            pickle.UnpicklingError: If model file is corrupted
        """
        import os
        
        # Ensure model name has .pkl extension by appending if needed
        model_name = f"{model_name}.pkl" if not model_name.endswith('.pkl') else model_name
        
        # Construct full path to model file in models directory
        model_path = os.path.join('models', model_name)
        
        # Create models directory if it doesn't exist yet
        # exist_ok=True prevents errors if directory already exists
        os.makedirs('models', exist_ok=True)
        
        try:
            # Open model file in binary read mode and deserialize using pickle
            with open(model_path, 'rb') as model_file:
                # Load the model object from the file
                model = pickle.load(model_file)
                # Log successful load
                self.logger.info(f"Model successfully loaded from {model_path}")
                return model
                
        except FileNotFoundError:
            # Handle case where model file doesn't exist
            self.logger.error(f"Model file not found at {model_path}")
            return None
        except pickle.UnpicklingError as e:
            # Handle corrupted or invalid pickle file
            self.logger.error(f"Error unpickling model from {model_path}: {str(e)}")
            return None
        except Exception as e:
            # Catch any other unexpected errors
            self.logger.error(f"Unexpected error loading model from {model_path}: {str(e)}")
            return None

    def fetch_flight_dataset(self, force_fetch: bool = False) -> pd.DataFrame:
        """
        Fetch flight dataset from Kaggle using kagglehub and prepare it for modeling.
        
        Returns:
            pd.DataFrame: Loaded and prepared dataset
            
        Flow:
        1. Check if dataset is already cached in memory
        2. If not cached, fetch from Kaggle and prepare it
        3. Store both original and prepared versions
        4. Return the prepared dataset
        """
        # First check if we already have the dataset loaded in memory or on the computer
        # This prevents unnecessary re-downloading and processing
        if self.flight_dataset is not None and not force_fetch:
            self.logger.info("Using cached flight dataset")
            return self.flight_dataset
        
        #check if the dataset is already saved on the computer
        if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'flight_dataset_original.csv')) and not force_fetch:
            self.logger.info("Using original saved flight dataset")
            self.original_flight_dataset = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'flight_dataset_original.csv'))
            df = self.original_flight_dataset.copy()
        if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'flight_dataset.csv')) and not force_fetch:
            self.logger.info("Using saved flight dataset")
            self.flight_dataset = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'flight_dataset.csv'))
            return self.flight_dataset.copy()


        # Only fetch and prepare if we haven't done so already
        # This is tracked by the has_flight_dataset flag
        if not self.has_flight_dataset or force_fetch:
            self.logger.info("Fetching flight dataset...")
            
            if self.original_flight_dataset is None:
                # Get dataset from Kaggle using environment variables for path/filename
                df = self.fetch_dataset(
                    os.environ['KAGGLE_FLIGHT_FILE_PATH'],
                    os.environ['KAGGLE_FLIGHT_FILE_NAME']
                )
            
            
            self.logger.info("Preparing flight dataset...")
            # Store original version before any preprocessing
            self.original_flight_dataset = df.copy()
            # Process the dataset using prepare_flight_dataset method
            prepared_df = self.prepare_flight_dataset(df)
            
            # Cache both the prepared dataset and set flag
            self.flight_dataset = prepared_df
            self.has_flight_dataset = True

        # Final validation check - dataset should never be None at this point
        # If it is, something went wrong in the fetch/prepare process
        if self.flight_dataset is None:
            raise ValueError("Failed to load flight dataset")
        
        # Save the flight dataset to a CSV file in the resources folder
        resources_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources')
        os.makedirs(resources_dir, exist_ok=True)
        
        csv_path = os.path.join(resources_dir, 'flight_dataset.csv')
        csv_path_original = os.path.join(resources_dir, 'flight_dataset_original.csv')
        self.logger.info(f"Saving flight dataset to {csv_path}")
        
        try:
            self.original_flight_dataset.to_csv(csv_path_original, index=False)
            self.flight_dataset.to_csv(csv_path, index=False)
            self.logger.info("Successfully saved flight dataset")
        except Exception as e:
            self.logger.error(f"Error saving flight dataset: {str(e)}")
            # Continue execution even if save fails

        return self.flight_dataset.copy()

    def fetch_dataset(self, file_path: str, file_name: str) -> pd.DataFrame:
        """
        Fetch dataset from Kaggle using kagglehub.
        
        Args:
            file_path (str): Path to the Kaggle dataset
            file_name (str): Name of the file to download
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If dataset cannot be found or downloaded
            pd.errors.EmptyDataError: If the CSV file is empty
        """
        load_dotenv()
        
        try:
            # Download dataset and construct full path
            download_path = kagglehub.dataset_download(file_path)
            full_path = os.path.join(download_path, file_name)
            
            # Load and return the dataset
            return pd.read_csv(full_path, engine='python')
            
        except Exception as e:
            self.logger.error(f"Error fetching dataset: {str(e)}")
            raise

    def train_flight_delay_model(self, model_type: str = 'logistic_regression', test_size: float = 0.2, force_train: bool = False, force_fetch: bool = False) -> Tuple[Any, Dict[str, float]]:
        """
        Orchestrate the complete process of retrieving data and training a flight delay prediction model.
        """
        self.logger.info("Starting flight delay model training process...")
        
        if self.flight_dataset is None:
            # Step 1: Fetch and prepare the dataset
            self.logger.info("Step 1: Fetching and preparing the dataset...")
            df = self.fetch_flight_dataset(force_fetch)
            self.logger.info(f"Dataset shape: {df.shape}")
            
            # Step 2: Prepare features and target
            self.logger.info("Step 2: Preparing features and target...")
            X = df.drop(columns=['DELAY_CATEGORY'])
            y = df['DELAY_CATEGORY']

            # Save the data for later use
            self.xdata = X.copy()   
            self.ydata = y.copy() 
        else:
            self.logger.info("Flight dataset is already loaded. Using the loaded dataset...")
            df = self.flight_dataset 
        
        # Log original features for reference
        self.logger.info(f"Features being used: {self.xdata.columns.tolist()}")

        # Step 3: Split the data with stratification
        self.logger.info(f"Step 3: Splitting data into train/test sets (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.xdata, self.ydata, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=self.ydata  # Ensure balanced split
        )
        self.logger.info(f"Training set shape: {X_train.shape}")
        self.logger.info(f"Test set shape: {X_test.shape}")
        
        # Step 4: Preprocess the data
        self.logger.info("Step 4: Preprocessing features...")
        
        # Handle missing values
        X_train = self._handle_missing_values(X_train)
        X_test = self._handle_missing_values(X_test)
        
        # Encode categorical features
        X_train_processed = self._encode_categorical_features(X_train.copy())
        X_test_processed = self._encode_categorical_features(X_test.copy())
        
        # Scale numerical features
        numerical_cols = X_train_processed.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            X_train_processed[numerical_cols] = self.scaler.fit_transform(X_train_processed[numerical_cols])
            X_test_processed[numerical_cols] = self.scaler.transform(X_test_processed[numerical_cols])

        # Step 5: Train the model
        self.logger.info(f"Step 5: Training {model_type} model...")
        model = self.train_model(X_train_processed, y_train, model_type=model_type, force_train=force_train)
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
        
    def _calculate_total_delay_by_day(self, day_of_year: int, airline: str, origin: str) -> float:
        """
        Calculate the mean total delay for a given day of the year from historical data.
        
        Args:
            day_of_year (int): Day of year (1-366)
            
        Returns:
            float: Mean total delay for that day of year
        """
        if self.original_flight_dataset is None:
            self.logger.warning("No flight dataset available for delay calculation")
            return 0.0
            
        # Convert FL_DATE to day of year
        df = self.original_flight_dataset.copy()
        df['DAY_OF_YEAR'] = pd.to_datetime(df['FL_DATE']).dt.dayofyear
        
        # Filter for the specific day
        day_data = df[df['DAY_OF_YEAR'] == day_of_year]
        
        if day_data.empty:
            self.logger.warning(f"No historical data found for day {day_of_year}")
            return 0.0
        
        # Calculate mean of delay columns for that day
        delay_columns = [
            'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 
            'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 
            'DELAY_DUE_LATE_AIRCRAFT'
        ]
        
        # Filter for specific airline and origin
        filtered_data = day_data[
            (day_data['AIRLINE'] == airline) & 
            (day_data['ORIGIN'] == origin)
        ]
        
        # If no data for specific airline/origin, fall back to overall average
        if filtered_data.empty:
            total_delay = day_data[delay_columns].mean().mean()
        else:
            total_delay = filtered_data[delay_columns].mean().mean()
        
        self.logger.debug(f"Calculated total delay for day {day_of_year}: {total_delay:.2f}")
        return total_delay

    def _is_holiday(self, date: str) -> int:
        """
        Check if a given date is a holiday.
        
        Args:
            date (str): Date in yyyy-mm-dd format
            
        Returns:
            int: 1 if the date is a holiday, 0 otherwise
        """
        holidays = [
            '2021-12-25', '2021-12-31', '2022-01-01', '2022-07-04', '2022-09-05',
            '2022-11-24', '2022-12-25', '2022-12-31', '2023-01-01', '2023-07-04',
            '2023-09-04', '2023-11-23', '2023-12-25', '2023-12-31', '2024-01-01',
            '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-25', '2024-12-31'
        ]
        return 1 if date in holidays else 0

    def _get_historical_delay(self, airline: str, origin: str) -> float:
        """
        Get historical average delay for a specific airline and origin airport.
        
        Args:
            airline (str): Airline name
            origin (str): Origin airport code
            
        Returns:
            float: Historical average delay
        """
        if self.original_flight_dataset is None:
            self.logger.warning("No flight dataset available for historical delay calculation")
            return 0.0
            
        # Filter data for specific airline and origin
        mask = (self.original_flight_dataset['AIRLINE'] == airline) & \
               (self.original_flight_dataset['ORIGIN'] == origin)
        
        filtered_data = self.original_flight_dataset[mask]
        
        if filtered_data.empty:
            self.logger.warning(f"No historical data found for {airline} at {origin}")
            return 0.0
            
        # Calculate average delay from all delay columns
        delay_columns = [
            'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 
            'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 
            'DELAY_DUE_LATE_AIRCRAFT'
        ]
        
        historical_delay = filtered_data[delay_columns].mean().mean()
        self.logger.debug(f"Historical delay for {airline} at {origin}: {historical_delay:.2f}")
        
        return historical_delay
