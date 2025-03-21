import pytest
import pandas as pd
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.helperclasses import DataFetcher
import requests
from datetime import datetime

@pytest.fixture
def mock_response():
    """Create a mock response object."""
    mock = Mock()
    mock.json.return_value = {
        'data': [
            {'id': 1, 'name': 'Test 1'},
            {'id': 2, 'name': 'Test 2'}
        ]
    }
    mock.raise_for_status.return_value = None
    return mock

@pytest.fixture
def data_fetcher():
    """Create a DataFetcher instance for testing."""
    return DataFetcher(
        base_url="https://api.example.com",
        api_key="test-api-key",
        rate_limit=60,
        timeout=30,
        retry_attempts=3,
        retry_delay=1
    )

def test_init(data_fetcher):
    """Test DataFetcher initialization."""
    assert data_fetcher.base_url == "https://api.example.com"
    assert data_fetcher.api_key == "test-api-key"
    assert data_fetcher.rate_limit == 60
    assert data_fetcher.timeout == 30
    assert data_fetcher.retry_attempts == 3
    assert data_fetcher.retry_delay == 1
    assert data_fetcher.request_times == []
    assert data_fetcher.request_count == 0

def test_make_request(data_fetcher, mock_response):
    """Test the _make_request method."""
    with patch('requests.request', return_value=mock_response) as mock_request:
        response = data_fetcher._make_request('test-endpoint')
        
        # Check if request was made with correct parameters
        mock_request.assert_called_once()
        call_args = mock_request.call_args[1]
        assert call_args['method'] == 'GET'
        assert call_args['url'] == "https://api.example.com/test-endpoint"
        assert call_args['headers']['Authorization'] == 'Bearer test-api-key'
        assert call_args['timeout'] == 30

def test_fetch_data_json(data_fetcher, mock_response):
    """Test fetch_data with JSON response format."""
    with patch('requests.request', return_value=mock_response) as mock_request:
        result = data_fetcher.fetch_data('test-endpoint', response_format='json')
        
        assert isinstance(result, dict)
        assert 'data' in result
        assert len(result['data']) == 2
        mock_request.assert_called_once()

def test_fetch_data_dataframe(data_fetcher, mock_response):
    """Test fetch_data with DataFrame response format."""
    with patch('requests.request', return_value=mock_response) as mock_request:
        result = data_fetcher.fetch_data('test-endpoint', response_format='dataframe')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'id' in result.columns
        assert 'name' in result.columns
        mock_request.assert_called_once()

def test_fetch_paginated_data(data_fetcher):
    """Test fetch_paginated_data method."""
    # Create mock responses for multiple pages
    mock_responses = [
        Mock(json=lambda: {'data': [{'id': 1}, {'id': 2}]}),
        Mock(json=lambda: {'data': [{'id': 3}, {'id': 4}]}),
        Mock(json=lambda: {'data': []})  # Empty response to end pagination
    ]
    
    with patch('requests.request', side_effect=mock_responses) as mock_request:
        result = data_fetcher.fetch_paginated_data(
            'test-endpoint',
            page_size=2,
            max_pages=2
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert mock_request.call_count == 2  # Should stop after 2 pages

def test_save_to_file_csv(data_fetcher, tmp_path):
    """Test saving data to CSV file."""
    test_data = pd.DataFrame({
        'id': [1, 2],
        'name': ['Test 1', 'Test 2']
    })
    
    file_path = tmp_path / "test.csv"
    data_fetcher.save_to_file(test_data, str(file_path), format='csv')
    
    assert file_path.exists()
    saved_data = pd.read_csv(file_path)
    assert len(saved_data) == 2
    assert all(col in saved_data.columns for col in ['id', 'name'])

def test_save_to_file_json(data_fetcher, tmp_path):
    """Test saving data to JSON file."""
    test_data = {'data': [{'id': 1}, {'id': 2}]}
    
    file_path = tmp_path / "test.json"
    data_fetcher.save_to_file(test_data, str(file_path), format='json')
    
    assert file_path.exists()
    with open(file_path) as f:
        saved_data = json.load(f)
    assert saved_data == test_data

def test_rate_limiting(data_fetcher, mock_response):
    """Test rate limiting functionality."""
    with patch('requests.request', return_value=mock_response) as mock_request:
        # Make multiple requests
        for _ in range(3):
            data_fetcher._make_request('test-endpoint')
        
        # Check rate limit tracking
        stats = data_fetcher.get_request_stats()
        assert stats['total_requests'] == 3
        assert stats['requests_last_minute'] == 3
        assert stats['rate_limit'] == 60

def test_retry_logic(data_fetcher):
    """Test retry logic for failed requests."""
    # Create a mock response that succeeds on the third try
    mock_success = Mock()
    mock_success.raise_for_status.return_value = None
    mock_success.json.return_value = {'data': [{'id': 1}]}
    
    def side_effect(*args, **kwargs):
        if side_effect.counter < 2:  # Fail first two times
            side_effect.counter += 1
            raise requests.exceptions.RequestException(f"Failure {side_effect.counter}")
        return mock_success
    
    side_effect.counter = 0
    
    with patch('requests.request', side_effect=side_effect) as mock_request:
        response = data_fetcher._make_request('test-endpoint')
        
        assert mock_request.call_count == 3  # Should have tried 3 times
        assert response.json() == {'data': [{'id': 1}]}  # Should get successful response

def test_invalid_response_format(data_fetcher, mock_response):
    """Test handling of invalid response format."""
    with patch('requests.request', return_value=mock_response) as mock_request:
        with pytest.raises(ValueError, match="Unsupported response format"):
            data_fetcher.fetch_data('test-endpoint', response_format='invalid')

def test_error_handling(data_fetcher):
    """Test error handling for failed requests."""
    with patch('requests.request', side_effect=requests.exceptions.RequestException("API Error")):
        with pytest.raises(requests.exceptions.RequestException):
            data_fetcher.fetch_data('test-endpoint')

def test_get_request_stats(data_fetcher, mock_response):
    """Test request statistics tracking."""
    with patch('requests.request', return_value=mock_response) as mock_request:
        # Make a request
        data_fetcher._make_request('test-endpoint')
        
        # Get stats
        stats = data_fetcher.get_request_stats()
        
        assert stats['total_requests'] == 1
        assert stats['requests_last_minute'] == 1
        assert stats['rate_limit'] == 60
        assert isinstance(stats['last_request_time'], str)
        assert datetime.fromisoformat(stats['last_request_time']) 