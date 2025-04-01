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
| Model/UI Connection | 3/25/25 | Complete |
| Testing | Ad Hoc | Complete |
| Finalize Documentation | 3/27/25 | Complete |
| Create Presentation | 3/27/25 | Complete |

## Proposal
Air travel: where the thrill of soaring 35,000 feet in the air is matched only by the agony of waiting on the ground for your delayed flight. It's like being a contestant in a real-life game of "Will I Make My Connection?"—except instead of prizes, you win stress and a possible overnight stay in an airport lounge.

Our proposal is to create a machine learning model that helps you predict whether your flight will be delayed. By feeding it three crucial pieces of information —date, origin, and airline— our model will give you a heads-up on whether you should start practicing your "I'm stuck in the airport" face or if you'll actually make it to your destination on time.

## Slide Deck
[Ready? Set? Wait.](https://docs.google.com/presentation/d/1pJ8xgxxK05_RRqku2S9UNKVRdpPeZABSz4UzP7nNt7k/edit?usp=sharing)

## Data sets
[Flight Data](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023)

## Future Research Questions and Issues
- During development, we discovered a gap in our Airplane Data for the year 2024. After assessing the situation, we determined that incorporating this missing data would exceed our project timeline. For future iterations, we recommend including the 2024 data to ensure the model is trained on the most current and comprehensive information available. This update would enhance the accuracy and reliability of our predictions.
- Due to the paywall limits of the NOAA weather, we were unable to pull all of the historical data we would have needed to integrate the weather into our model. Future iterations would see us using that data.
- During development we found that there was some awesome categorical data within the kaggle set such as the reason why a flight was delayed that would be interesting to utilize. The time restrictions for this project and hardware memory limits made more nuanced and complicated model training impossible.
- Found that due to improvements within the airline industry delayed flights are uncommon. We would love to look into adapting this model to a neural network in the future to handle multiple targets to see if cancellations or diverted flights influence the probablilities.

# Program Information

## Overview
This program is a flight delay prediction system that uses machine learning to forecast whether flights will be delayed based on historical flight data. It provides both a model training pipeline and prediction capabilities through a simple API interface.

### Core Features
- Flight delay prediction using airline, date, and origin airport
- Support for multiple ML models (Random Forest, Logistic Regression, XGBoost)
- Automated data preprocessing and feature engineering
- Model performance evaluation and metrics tracking
- Efficient handling of large flight datasets
- Built-in data validation and error handling
- Model persistence and loading capabilities
- Comprehensive logging system

### Programming Languages
- Python 3.11+

### Required Libraries/Dependencies

To install required files us the command 'pip install -r requirements.txt'

All dependencies are listed in requirements.txt and include:
- pandas>=2.1.0 - Data manipulation and analysis
- numpy>=1.24.0 - Numerical computing
- scikit-learn>=1.3.0 - Machine learning algorithms
- imbalanced-learn>=0.11.0 - Handling imbalanced datasets
- xgboost>=2.0.0, lightgbm>=4.1.0 - Gradient boosting implementations
- matplotlib>=3.7.0, seaborn>=0.12.0 - Data visualization
- kagglehub[pandas-datasets]>=0.1.0 - Kaggle dataset access
- python-dotenv>=1.0.0 - Environment variable management
- pytest>=7.4.0 - Testing framework
- tkinter>=8.6.0, tkcalendar>=1.6.0 - GUI development and calendar widget 

### Development Environment
- Python 3.11+
- Git for version control
- Virtual environment (recommended)
- Jupyter notebooks for analysis and visualization

### Model Results
- Random Forest:
Accuracy: 0.7708
Precision: 0.7877
Recall: 0.7708
F1: 0.7749

- Logistic Regression:
Accuracy: 0.6709
Precision: 0.6843
Recall: 0.6709
F1: 0.6758
