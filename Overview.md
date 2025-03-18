# Project 2 Overview

## Project Requirements

For Project 2, you will work with your group to solve, analyze, or visualize a problem using machine learning (ML), along with the other technologies you've learned so far. Here are the specific requirements:

- Find a problem worth solving, analyzing, or visualizing.
- The dataset(s) for your project must have at least 500 records. (If developing a decision tree/random forest model, you should use at least 1,000 records.)
- Use either a supervised or unsupervised ML model to solve, analyze, or visualize the problem.
- Evaluate the trained model(s) using testing data. Include any calculations, metrics, or visualizations needed to evaluate the performance.
- You must use Scikit-learn.
- You must use at least three of the following:
  - API requests
  - Matplotlib
  - Pandas
  - Pandas plotting
  - Prophet
  - Python
  - Time series analysis

## Industry Focus Examples

### Finance
- Create an algorithm that analyzes credit scores and predicts consumer personal-loan eligibility.
- Train an algorithm to analyze consumer spending and predict trends.
- Train an image classifier to assess property value, which could then be used to calculate insurance quotes.

### Healthcare
- Train an algorithm to recognize disease symptoms and predict if a patient is at risk.
- Train an image classifier to recognize anomalies, such as suspicious vs healthy areas of skin.
- Create an algorithm to analyze patient history and predict the likelihood of inherited illness.

### Custom
We've only specified healthcare and finance, but any industry can benefit from machine learning. Consider preparing a data deep dive or infrastructure review that shows machine learning in the context of what we've already learned.

- Create a front-end interface that maps to an API to "smarten" the algorithm.
- Perform a deep dive on existing data using machine learning.
- Create a visualization that continues to learn where clusters lie based on ML (use Leaflet or Plotly to change the visualization).
- Create an idea using mock data, and simulate how machine learning might be used.
- Create an analysis of existing data to make a prediction, classification, or regression.

## Working with Your Group

When working on an online group project, it's crucial to meet with your group and communicate regularly. Plan for significant collaboration time outside of class. The following tips can help you make the most of your time:

1. Decide how you're going to communicate with your group members when you begin. Create a Slack channel, exchange phone numbers, and ensure that the group knows each group member's available working hours.
2. Set up an agile project by using GitHub Projects so that your group can track tasks.
3. Create internal milestones to ensure that your group is on track. Set due dates for these milestones so that you have a timeline for completing the project. Some of these milestones might include:
   - Project ideation
   - Data fetching
   - Data exploration
   - Data transformation
   - Data analysis
   - Testing
   - Creating documentation
   - Creating the presentation

Since this is a two-week project, make sure that you have completed at least half of your project by the end of the first week in order to stay on track.

Although you will divide the work among the group members, it's essential to collaborate and communicate while working on different parts of the project. Be sure to check in with your teammates regularly and offer support.

## Support and Resources

Your instructional team will provide support during classes and office hours. You will also have access to learning assistants and tutors to help you with topics as needed. Make sure to take advantage of these resources as you collaborate with your group on this project.

## Requirements

### Data Model Implementation (25 points)
- There is a Jupyter notebook that thoroughly describes the data extraction, cleaning, and transformation process, and the cleaned data is exported as CSV files for the machine learning model. (10 points)
- A Python script initializes, trains, and evaluates a model or loads a pretrained model. (10 points)
- The model demonstrates meaningful predictive power at least 75% classification accuracy or 0.80 R-squared. (5 points)

### Data Model Optimization (25 points)
- The model optimization and evaluation process showing iterative changes made to the model and the resulting changes in model performance is documented in either a CSV/Excel table or in the Python script itself. (15 points)
- Overall model performance is printed or displayed at the end of the script. (10 points)

### GitHub Documentation (25 points)
- GitHub repository is free of unnecessary files and folders and has an appropriate .gitignore in use. (10 points)
- The README is customized as a polished presentation of the content of the project. (15 points)

### Presentation Requirements (25 points)
Your presentation should cover the following:

- An executive summary or overview of the project and project goals. (5 points)
- An overview of the data collection, cleanup, and exploration processes. Include a description of how you evaluated the trained model(s) using testing data. (5 points)
- The approach that your group took in achieving the project goals. (5 points)
- Any additional questions that surfaced, what your group might research next if more time was available, or share a plan for future development. (3 points)
- The results and conclusions of the application or analysis. (3 points)
- Slides effectively demonstrate the project. (2 points)
- Slides are visually clean and professional. (2 points)

## Grading Scale

| Grade | Points |
|-------|---------|
| A (+/-) | 90+ |
| B (+/-) | 80–89 |
| C (+/-) | 70–79 |
| D (+/-) | 60–69 |
| F (+/-) | < 60 |

## Project Guidelines

The following project guidelines focus on teamwork, your project proposal, data sources, and data cleanup and analysis.

### Collaborating with Your Team

Remember that these projects are a group effort. The experience of close collaboration will create better project outcomes and help you in your future careers. Specifically, you'll learn collaborative workflows that will enable you to approach and solve complex problems. Working in groups allows you to work smart and dream big. Take advantage!

### Project Proposal

Before you start writing any code, your group should outline the scope and purpose of your project. This will help provide direction and safeguard against scope creep (the tendency for projects to become more complex after work begins).

The proposal is essentially a brief summary of your interests and intent. Be sure to include the following details:

- The kind of data you'd like to work with and the field you're interested in (finance, healthcare, etc.)
- The questions you'll ask of the data
- A possible source for the data

Use the following example for guidance:

> The aim of our project is to uncover patterns in credit card fraud. We'll examine relationships between transaction types and location, purchase prices and times of day, purchase trends over the course of a year, and other related relationships derived from the data.

### Finding Data

Once your group has written a proposal, it's time to start searching for data. We recommend the following curated sources of high-quality data:

- [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [data.world](https://data.world/)
- [Kaggle](https://www.kaggle.com/)
- [Data.gov](https://www.data.gov/)
- [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets)

**Important**: Whenever you use a dataset or create a new dataset based on other sources (such as existing datasets or information scraped from websites), make sure to use the following guidelines:

1. Check for copyright protections, and make sure that the way you plan to use this dataset is within the bounds of fair use.
2. Document how you intend to use this dataset now and in the future. Find any licenses or terms of use associated with the dataset, and review them to confirm that your intended use is in compliance.
3. Investigate how the dataset was collected. Identify any indicators that the data was obtained from a source that the compilers were not authorized to access.

You'll likely have to adjust your project plan as you explore the available data. That's okay! This is all part of the process. Just make sure that everyone in the group is aligned on the project's goals as you make changes.

Make sure that your datasets are not too large for your personal computer. Big datasets are difficult to manage locally, so consider using data subsets or different datasets altogether.

### Data Cleanup and Analysis

Now that you've picked your data, it's time to tackle development and analysis. This is where the fun starts!

The analysis process can be broken into two broad phases: (1) Exploration and cleanup, and (2) analysis.

As you've learned, you'll need to explore, clean, and reformat your data before you can begin answering your research questions. We recommend keeping track of these exploration and cleanup steps in a dedicated Jupyter notebook to stay organized and make it easier to present your work later.

After you've cleaned your data and are ready to start crunching numbers, you should track your work in a Jupyter notebook dedicated specifically to analysis. We recommend focusing your analysis on multiple techniques, such as aggregation, correlation, comparison, summary statistics, sentiment analysis, and time-series analysis. Don't forget to include plots during both the exploration and analysis phases. Creating plots along the way can reveal insights and interesting trends in the data that you might not notice if you wait until you're preparing for your presentation.

## Presentation Guidelines

This section lists the Project 2 presentation guidelines. Each group will prepare a formal, 10-minute presentation (7 minutes for the presentation followed by a 3-minute question-and-answer session) that covers the following points.

1. **An executive summary or overview of the project and project goals**:
   - Explain how the project relates to the industry you selected.

2. **An overview of the data collection, cleanup, and exploration processes**:
   - Describe the source of your data and why you chose it for your project.
   - Describe the collection, exploration, and cleanup process.

3. **The approach that your group took to achieve the project goals**:
   - Include any relevant code or demonstrations of the application or analysis.
   - Discuss any unanticipated insights or problems that arose and how you resolved them.

4. **The results/conclusions of the application or analysis**:
   - Include relevant images or examples to support your work.
   - If the project goal was not achieved, discuss the issues and how you attempted to resolve them.

5. **Next steps**:
   - Briefly discuss potential next steps for the project.

It's crucial that you find time to rehearse before presentation day.

On the day of your presentation, each member of your group is required to submit the URL of your GitHub repository for grading.

## Presentation Day

Your group will have a total of 10 minutes—7 minutes for the presentation followed by a 3-minute question-and-answer session. It's crucial that you find time to rehearse before presentation day.

On the day of your presentation, each member of your group is required to submit the URL of your GitHub repository for grading.
