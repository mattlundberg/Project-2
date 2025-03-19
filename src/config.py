import os
from typing import Dict, Any
from dotenv import load_dotenv
import logging

class Config:
    """Configuration manager for the application."""
    
    def __init__(self, env_file: str = '.env'):
        """
        Initialize configuration from environment variables.
        
        Args:
            env_file (str): Path to the .env file
        """
        # Load environment variables from .env file
        load_dotenv(env_file)
        
        # API Configuration
        self.api_base_url = os.getenv('API_BASE_URL', 'https://api.example.com')
        self.api_key = os.getenv('API_KEY')
        
        # Rate Limiting
        self.api_rate_limit = int(os.getenv('API_RATE_LIMIT', '60'))
        self.api_timeout = int(os.getenv('API_TIMEOUT', '30'))
        self.api_retry_attempts = int(os.getenv('API_RETRY_ATTEMPTS', '3'))
        self.api_retry_delay = int(os.getenv('API_RETRY_DELAY', '5'))
        
        # Logging Configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_file = os.getenv('LOG_FILE', 'api_fetcher.log')
        
        # Data Storage
        self.data_output_dir = os.getenv('DATA_OUTPUT_DIR', 'data')
        self.default_file_format = os.getenv('DEFAULT_FILE_FORMAT', 'csv')
        
        # Proxy Configuration
        self.http_proxy = os.getenv('HTTP_PROXY')
        self.https_proxy = os.getenv('HTTPS_PROXY')
        
        # Setup logging
        self._setup_logging()
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_output_dir, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging based on environment settings."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
    
    def get_proxy_settings(self) -> Dict[str, str]:
        """
        Get proxy settings as a dictionary.
        
        Returns:
            Dict[str, str]: Proxy settings for requests
        """
        proxies = {}
        if self.http_proxy:
            proxies['http'] = self.http_proxy
        if self.https_proxy:
            proxies['https'] = self.https_proxy
        return proxies
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return {
            'api_base_url': self.api_base_url,
            'api_key': '***' if self.api_key else None,  # Mask API key
            'api_rate_limit': self.api_rate_limit,
            'api_timeout': self.api_timeout,
            'api_retry_attempts': self.api_retry_attempts,
            'api_retry_delay': self.api_retry_delay,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'data_output_dir': self.data_output_dir,
            'default_file_format': self.default_file_format,
            'proxies': self.get_proxy_settings()
        }
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            bool: True if configuration is valid
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.api_base_url:
            raise ValueError("API_BASE_URL is required")
            
        if self.api_rate_limit < 1:
            raise ValueError("API_RATE_LIMIT must be greater than 0")
            
        if self.api_timeout < 1:
            raise ValueError("API_TIMEOUT must be greater than 0")
            
        if self.api_retry_attempts < 0:
            raise ValueError("API_RETRY_ATTEMPTS must be non-negative")
            
        if self.api_retry_delay < 0:
            raise ValueError("API_RETRY_DELAY must be non-negative")
            
        return True

# Create a global configuration instance
config = Config() 