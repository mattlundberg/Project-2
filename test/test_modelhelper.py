import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import sys
import os

# Here is the command to run the tests
# pytest test/test_modelhelper.py -v

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from modelhelper import ModelHelper

@pytest.fixture
def sample_classification_data():
    """Create sample classification data for testing."""
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                             n_redundant=5, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    df['target'] = y
    return df

@pytest.fixture
def sample_regression_data():
    """Create sample regression data for testing."""
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    df['target'] = y
    return df

@pytest.fixture
def sample_data_with_missing():
    """Create sample data with missing values for testing."""
    df = pd.DataFrame({
        'feature_1': [1, 2, np.nan, 4, 5],
        'feature_2': [10, np.nan, 30, 40, 50],
        'feature_3': ['A', 'B', 'C', 'D', 'E'],
        'target': [0, 1, 0, 1, 0]
    })
    return df

@pytest.fixture
def model_helper():
    """Create a ModelHelper instance for testing."""
    return ModelHelper(random_state=42)

def test_clean_data(model_helper, sample_data_with_missing):
    """Test the clean_data method."""
    # Test cleaning data with missing values
    cleaned_df = model_helper.clean_data(sample_data_with_missing, 'target')
    
    # Check if missing values are handled
    assert not cleaned_df.isnull().any().any()
    
    # Check if categorical columns are preserved
    assert 'feature_3' in cleaned_df.columns
    
    # Check if target column is preserved
    assert 'target' in cleaned_df.columns

def test_preprocess_features(model_helper, sample_data_with_missing):
    """Test the preprocess_features method."""
    # Test preprocessing with scaling
    X, y = model_helper.preprocess_features(sample_data_with_missing, 'target', scale_data=True)
    
    # Check if categorical variables are encoded
    assert X['feature_3'].dtype in ['int32', 'int64']
    
    # Check if numerical features are scaled
    assert np.isclose(X['feature_1'].mean(), 0, atol=1e-10)
    assert np.isclose(X['feature_1'].std(), 1, atol=1e-10)
    
    # Check if target is preserved
    assert len(y) == len(sample_data_with_missing)

def test_prepare_data(model_helper, sample_classification_data):
    """Test the prepare_data method."""
    # Test data preparation
    X_train, X_test, y_train, y_test = model_helper.prepare_data(
        sample_classification_data, 'target', test_size=0.2
    )
    
    # Check shapes
    assert len(X_train) + len(X_test) == len(sample_classification_data)
    assert len(y_train) + len(y_test) == len(sample_classification_data)
    
    # Check if data is scaled
    assert np.isclose(X_train.mean().mean(), 0, atol=1e-10)
    assert np.isclose(X_train.std().mean(), 1, atol=1e-10)

def test_train_model_classification(model_helper, sample_classification_data):
    """Test model training for classification."""
    # Prepare data
    X_train, X_test, y_train, y_test = model_helper.prepare_data(
        sample_classification_data, 'target'
    )
    
    # Test different classification models
    for model_type in ['random_forest', 'logistic_regression', 'svm']:
        model = model_helper.train_model(
            X_train, y_train, model_type=model_type, task='classification'
        )
        assert model is not None
        assert hasattr(model, 'predict')

def test_train_model_regression(model_helper, sample_regression_data):
    """Test model training for regression."""
    # Prepare data
    X_train, X_test, y_train, y_test = model_helper.prepare_data(
        sample_regression_data, 'target'
    )
    
    # Test different regression models
    for model_type in ['random_forest', 'linear_regression', 'svr']:
        model = model_helper.train_model(
            X_train, y_train, model_type=model_type, task='regression'
        )
        assert model is not None
        assert hasattr(model, 'predict')

def test_evaluate_model_classification(model_helper, sample_classification_data):
    """Test model evaluation for classification."""
    # Prepare and train data
    X_train, X_test, y_train, y_test = model_helper.prepare_data(
        sample_classification_data, 'target'
    )
    model = model_helper.train_model(
        X_train, y_train, model_type='random_forest', task='classification'
    )
    
    # Evaluate model
    metrics = model_helper.evaluate_model(model, X_test, y_test, task='classification')
    
    # Check metrics
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1

def test_evaluate_model_regression(model_helper, sample_regression_data):
    """Test model evaluation for regression."""
    # Prepare and train data
    X_train, X_test, y_train, y_test = model_helper.prepare_data(
        sample_regression_data, 'target'
    )
    model = model_helper.train_model(
        X_train, y_train, model_type='random_forest', task='regression'
    )
    
    # Evaluate model
    metrics = model_helper.evaluate_model(model, X_test, y_test, task='regression')
    
    # Check metrics
    assert 'mse' in metrics
    assert 'r2' in metrics
    assert metrics['mse'] >= 0
    assert -1 <= metrics['r2'] <= 1

def test_predict(model_helper, sample_classification_data):
    """Test the predict method."""
    # Prepare and train data
    X_train, X_test, y_train, y_test = model_helper.prepare_data(
        sample_classification_data, 'target'
    )
    model = model_helper.train_model(
        X_train, y_train, model_type='random_forest', task='classification'
    )
    
    # Make predictions
    predictions = model_helper.predict(model, X_test)
    
    # Check predictions
    assert len(predictions) == len(X_test)
    assert all(isinstance(pred, (int, np.integer)) for pred in predictions)

def test_invalid_model_type(model_helper, sample_classification_data):
    """Test handling of invalid model type."""
    # Prepare data
    X_train, X_test, y_train, y_test = model_helper.prepare_data(
        sample_classification_data, 'target'
    )
    
    # Test invalid model type
    with pytest.raises(ValueError):
        model_helper.train_model(
            X_train, y_train, model_type='invalid_model', task='classification'
        )

def test_invalid_task(model_helper, sample_classification_data):
    """Test handling of invalid task type."""
    # Prepare data
    X_train, X_test, y_train, y_test = model_helper.prepare_data(
        sample_classification_data, 'target'
    )
    
    # Test invalid task
    with pytest.raises(ValueError):
        model_helper.train_model(
            X_train, y_train, model_type='random_forest', task='invalid_task'
        )

def test_run_full_pipeline(model_helper, sample_classification_data, sample_regression_data):
    """Test running the full model pipeline for both classification and regression."""
    # Test classification pipeline
    X_train_c, X_test_c, y_train_c, y_test_c = model_helper.prepare_data(
        sample_classification_data, 'target'
    )
    
    clf_model = model_helper.train_model(
        X_train_c, y_train_c, model_type='random_forest', task='classification'
    )
    
    clf_metrics = model_helper.evaluate_model(
        clf_model, X_test_c, y_test_c, task='classification'
    )
    
    # Verify classification results
    assert 'accuracy' in clf_metrics
    assert 0 <= clf_metrics['accuracy'] <= 1
    
    # Test regression pipeline
    X_train_r, X_test_r, y_train_r, y_test_r = model_helper.prepare_data(
        sample_regression_data, 'target'
    )
    
    reg_model = model_helper.train_model(
        X_train_r, y_train_r, model_type='random_forest', task='regression'
    )
    
    reg_metrics = model_helper.evaluate_model(
        reg_model, X_test_r, y_test_r, task='regression'
    )
    
    # Verify regression results
    assert 'mse' in reg_metrics
    assert 'r2' in reg_metrics
    assert reg_metrics['mse'] >= 0
    assert -1 <= reg_metrics['r2'] <= 1