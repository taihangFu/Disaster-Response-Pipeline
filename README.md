# Disaster-Response-Pipeline

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [Instructions](#instructions)
3. [Examples](#examples)

<a name="descripton"></a>
## Description

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with [Figure Eight](https://www.figure-eight.com/).
The initial dataset contains pre-labelled tweet and messages from real-life disaster. 
The goal of the project is to analyze disaster data from Figure Eight and build a model tt classify disaster messages.

The Project is divided in the following Sections:

1. ETL Pipeline to extract data from source, clean data and save into SQLlite database 
2. Machine Learning Pipeline to load data from database and train a classifier for messages
3. Web App where a user or an emergency worker can input a new message and get classification results in several categories.(Examples are provided on Screenshot section)

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Dependencies
* Python 3.5+ 
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

<a name="instructions"></a>
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<a name="examples"></a>
## Examples

Enter text message on input field

![](media/disaster-response-project1.png)



then the app will classify the input text meaasge and display the classification result.

![](media/disaster-response-project2-2.png)





