# Project-2: Ready? Set? Wait.
## Slack Channel
Join our Slack channel for project discussions and updates:
- Channel: #404-not-found
- Link: [404 Not Found](https://aiptwestnovem-cki2893.slack.com/archives/C089LSTUQER)

## Team Members
- Tiffany Jimenez
- Sam Lara
- Matthew Lundberg
- Jason Smoody
- Erin Spencer-Priebe 
 
 ## Project Milestones

| Milestone | Due Date | Status |
|----------|----------|----------|
| Project Ideation | 3/18/25 | Complete |
| Data Fetching & Clean Up | 3/20/25 | Complete |
| Data Exploration & Build | 3/24/25 | Complete |
| Model Selection | 3/25/25 | Complete |
| Model/UI Connection | 3/25/25 | In Progress |
| Testing | Ad Hoc | In Progress |
| Finalize Documentation | 3/27/25 | To Do |
| Create Presentation | 3/27/25 | To Do |

## Proposal
Air travel: where the thrill of soaring 35,000 feet in the air is matched only by the agony of waiting on the ground for your delayed flight. It's like being a contestant in a real-life game of "Will I Make My Connection?"—except instead of prizes, you win stress and a possible overnight stay in an airport lounge.

Our proposal is to create a machine learning model that helps you predict whether your flight will be delayed. By feeding it three crucial pieces of information —date, origin, and airline— our model will give you a heads-up on whether you should start practicing your "I'm stuck in the airport" face or if you'll actually make it to your destination on time.

## Slide Deck
[Ready? Set? Wait.](https://docs.google.com/presentation/d/1pJ8xgxxK05_RRqku2S9UNKVRdpPeZABSz4UzP7nNt7k/edit?usp=sharing)

## Data sets
[Airplane Data](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023)

## Future Research Questions and Issues
- During development, we discovered a gap in our Airplane Data for the year 2024. After assessing the situation, we determined that incorporating this missing data would exceed our project timeline. For future iterations, we recommend including the 2024 data to ensure the model is trained on the most current and comprehensive information available. This update would enhance the accuracy and reliability of our predictions.
- Due to the paywall limits of the NOAA weather, we were unable to pull all of the historical data we would have needed to integrate the weather into our model. Future iterations would see us using that data.
- During development we found that there was some awesome categorical data within the kaggle set such as the reason why a flight was delayed that would be interesting to utilize. The time restrictions for this project and hardware memory limits made more nuanced and complicated model training impossible.

# Program Information 
## Program Information
- TBD

### Programming Languages
- Python 3.x

### Required Libraries/Dependencies
- requests>=2.31.0
- pandas>=2.1.0
- python-dotenv>=1.0.0
- pytest>=7.4.0
- openpyxl>=3.1.2 (for Excel support)

### Development Environment
- Python 3.x
- Git for version control
- Virtual environment (recommended)

### Project Structure
```
project/
├── src/
│   ├── __init__.py
│   ├── helperclasses.py    # Data fetching classes
│   ├── modelhelper.py      # Machine learning helper
│   └── config.py          # Configuration management
├── test/
│   ├── __init__.py
│   ├── test_helperclasses.py
│   └── test_modelhelper.py
├── data/                  # Directory for downloaded data
├── .env                   # Environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

### Data Fetching Classes

#### DataFetcherAPI
A class for fetching and processing data from APIs with built-in error handling, rate limiting, and data validation.

Features:
- Rate limiting support
- Automatic retry logic
- Multiple response formats (JSON, DataFrame, List)
- Pagination support
- File saving capabilities
- Request statistics tracking

Example usage:
```python
from src.helperclasses import DataFetcherAPI

# Initialize the fetcher
fetcher = DataFetcherAPI(
    base_url='https://api.example.com',
    api_key='your_api_key',
    rate_limit=60
)

# Fetch data
data = fetcher.fetch_data(
    endpoint='users',
    response_format='dataframe'
)

# Save to file
fetcher.save_to_file(data, 'users.csv', format='csv')
```

#### DataFetcherCurl
A class for fetching data using cURL commands with built-in error handling and rate limiting.

Features:
- Execute cURL commands with retry logic
- Support for various output formats
- File download capabilities
- Rate limiting
- Comprehensive error handling
- Request statistics tracking

Example usage:
```python
from src.helperclasses import DataFetcherCurl

# Initialize the fetcher
fetcher = DataFetcherCurl(
    output_dir='data',
    rate_limit=60
)

# Fetch data using cURL
data = fetcher.fetch_data(
    curl_command='curl -H "Authorization: Bearer token" https://api.example.com/data',
    output_format='json'
)

# Download a file
file_path = fetcher.fetch_file(
    curl_command='curl https://example.com/file.zip',
    output_file='file.zip'
)
```

### Overview and Analysis
#### Proposal
TBD

