import requests
import pandas as pd
import time
from typing import Union, Dict, Any, Optional, List
import logging
from datetime import datetime
import json

class DataFetcher:
    """
    A class for fetching and processing data from APIs with built-in error handling,
    rate limiting, and data validation.
    """
    
    def __init__(self, 
                 base_url: str,
                 api_key: Optional[str] = None,
                 rate_limit: int = 60,  # requests per minute
                 timeout: int = 30,
                 retry_attempts: int = 3,
                 retry_delay: int = 5):
        """
        Initialize the DataFetcher with API configuration.
        
        Args:
            base_url (str): Base URL for the API
            api_key (str, optional): API key for authentication
            rate_limit (int): Maximum number of requests per minute
            timeout (int): Request timeout in seconds
            retry_attempts (int): Number of retry attempts for failed requests
            retry_delay (int): Delay between retry attempts in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.last_request_time = 0
        self.request_count = 0
        self.request_times = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _make_request(self, 
                     endpoint: str, 
                     method: str = 'GET',
                     params: Optional[Dict] = None,
                     data: Optional[Dict] = None,
                     headers: Optional[Dict] = None) -> requests.Response:
        """
        Make an API request with rate limiting and retry logic.
        
        Args:
            endpoint (str): API endpoint
            method (str): HTTP method (GET, POST, etc.)
            params (dict, optional): Query parameters
            data (dict, optional): Request body data
            headers (dict, optional): Request headers
            
        Returns:
            requests.Response: API response
            
        Raises:
            requests.exceptions.RequestException: If request fails after retries
        """
        # Ensure rate limiting
        self._check_rate_limit()
        
        # Prepare headers
        if headers is None:
            headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        # Prepare URL
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Make request with retry logic
        for attempt in range(self.retry_attempts):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Update rate limiting tracking
                self._update_rate_limit_tracking()
                
                return response
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed, attempt {attempt + 1}/{self.retry_attempts}: {str(e)}")
                
                if attempt == self.retry_attempts - 1:
                    self.logger.error(f"Request failed after {self.retry_attempts} attempts: {str(e)}")
                    raise
                
                time.sleep(self.retry_delay)
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # If we've hit the rate limit, wait
        if len(self.request_times) >= self.rate_limit:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached. Waiting {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
    
    def _update_rate_limit_tracking(self):
        """Update rate limit tracking after successful request."""
        current_time = time.time()
        self.request_times.append(current_time)
    
    def fetch_data(self, 
                  endpoint: str,
                  params: Optional[Dict] = None,
                  method: str = 'GET',
                  data: Optional[Dict] = None,
                  response_format: str = 'json') -> Union[pd.DataFrame, Dict, List]:
        """
        Fetch data from the API and return it in the specified format.
        
        Args:
            endpoint (str): API endpoint
            params (dict, optional): Query parameters
            method (str): HTTP method
            data (dict, optional): Request body data
            response_format (str): Desired output format ('json', 'dataframe', 'list')
            
        Returns:
            Union[pd.DataFrame, Dict, List]: API response in specified format
        """
        try:
            response = self._make_request(
                endpoint=endpoint,
                method=method,
                params=params,
                data=data
            )
            
            response_data = response.json()
            
            if response_format == 'json':
                return response_data
            elif response_format == 'dataframe':
                # Handle nested data structure
                if isinstance(response_data, dict) and 'data' in response_data:
                    return pd.DataFrame(response_data['data'])
                return pd.DataFrame(response_data)
            elif response_format == 'list':
                if isinstance(response_data, dict) and 'data' in response_data:
                    return response_data['data']
                return response_data
            else:
                raise ValueError(f"Unsupported response format: {response_format}")
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch data: {str(e)}")
            raise
    
    def fetch_paginated_data(self,
                           endpoint: str,
                           page_size: int = 100,
                           max_pages: Optional[int] = None,
                           params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Fetch paginated data from the API and combine all pages.
        
        Args:
            endpoint (str): API endpoint
            page_size (int): Number of items per page
            max_pages (int, optional): Maximum number of pages to fetch
            params (dict, optional): Additional query parameters
            
        Returns:
            pd.DataFrame: Combined data from all pages
        """
        all_data = []
        page = 1
        
        while True:
            if max_pages and page > max_pages:
                break
                
            # Add pagination parameters
            pagination_params = {
                'page': page,
                'per_page': page_size,
                **(params or {})
            }
            
            try:
                response = self._make_request(endpoint, params=pagination_params)
                response_data = response.json()
                
                # Handle nested data structure
                if isinstance(response_data, dict) and 'data' in response_data:
                    data = response_data['data']
                else:
                    data = response_data
                
                # Check if we've reached the end of the data
                if not data or len(data) == 0:
                    break
                    
                all_data.extend(data)
                page += 1
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Failed to fetch page {page}: {str(e)}")
                break
        
        return pd.DataFrame(all_data)
    
    def save_to_file(self,
                    data: Union[pd.DataFrame, Dict, List],
                    filename: str,
                    format: str = 'csv') -> None:
        """
        Save fetched data to a file.
        
        Args:
            data: Data to save
            filename (str): Output filename
            format (str): Output format ('csv', 'json', 'excel')
        """
        try:
            if format == 'csv' and isinstance(data, pd.DataFrame):
                data.to_csv(filename, index=False)
            elif format == 'json':
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=4)
            elif format == 'excel' and isinstance(data, pd.DataFrame):
                data.to_excel(filename, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            self.logger.info(f"Data saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save data: {str(e)}")
            raise
    
    def get_request_stats(self) -> Dict[str, Any]:
        """
        Get statistics about API requests.
        
        Returns:
            Dict containing request statistics
        """
        return {
            'total_requests': len(self.request_times),
            'requests_last_minute': len([t for t in self.request_times if time.time() - t < 60]),
            'rate_limit': self.rate_limit,
            'last_request_time': datetime.fromtimestamp(self.request_times[-1]).isoformat() if self.request_times else None
        }
