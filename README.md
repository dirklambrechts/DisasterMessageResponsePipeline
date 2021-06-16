# Disaster Message Response Pipeline

## Table of Contents
1. [Description](#description)
2. [Using the application](#getting_started)
	1. [Requirements to run project](#dependencies)
	2. [Installing](#installation)
	3. [Executing Program](#execution)
	4. [Additional Material](#material)
3. [Authors](#authors)



<a name="descripton"></a>
## Description

This is a project for the Udacity Nano degree in data science. The data provided included disaster response message provided by Figure Eight in collaboration with Udacity. The nature of the data set is a large number of actual messages sent to a disaster response messaging service. Each of these messages have been classfied according to a message category defined by Figure Eight. The objective of this project is to use the data set provided to build a model able to predict the category of a new message submitted. 

The scope of the project includes deploying a web application to take a message as an input and to return the predicted message category. The project consists of four main tasks:
1. Pre-processing the data and building an ETL (extract, transform and load) pipeline that stores the data in a SQLite database
2. Building a machine learning pipeline that trains on classification categories provided
3. Implement the ETL and machine learning pipeline into python scripts
4. Deploy the machine learning model into a web application able to predict the categories of new messages in real time

<a name="getting_started"></a>
## Using the application

<a name="dependencies"></a>
### Requirements to run project
* Python 3.6+
* Data manipulation: Pandas, Numpy
* Machine Learning Libraries: Scikit-Learn
* Natural Language Processing Libraries: NLTK
* Database management: SQLalchemy
* Model Loading and Saving Library: Pickle
* Web App and Data Visualization: Flask, Plotly

<a name="installation"></a>
### Installation
To clone the git repository:
```
git clone https://github.com/dirklambrechts/DisasterMessageResponsePipeline
```
<a name="execution"></a>
### Running the program:
1. Run the following commands to pre-process the data, train the model and deploy the web application:

    - To run the the ETL program the following command is used (the last argument in this command is the name for the database):
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponseDB`
    - To train the model that uses data from the database created in the previous step run the following command:
        `python models/train_classifier.py data/DisasterResponseDB.db models/classifier.pkl`

2. Run the following command to run the python script responsible for deploying the web application.
    `python run.py`

3. Go to http://0.0.0.0:3001/ to view and use the web application

<a name="material"></a>
### Additional Information

There are two jupyter notebook (.ipynb) files that can assist in describing the data pre-processing and modelling process in more detail. They are in the folder called data and models respectively. 

<a name="authors"></a>
## Authors

* [Dirk Lambrechts](https://github.com/dirklambrechts)
