# Disaster Response Pipeline Project

## Table of Contents
 * [Project Motivation](#project-motivation)
 * [File Descriptions](#file-descriptions)
 * [Objective](#objective)
 * [Components](#components)
 * [Instructions on How to Interact With the Project](#instructions-of-how-to-interact-with-project)
 
### Project Motivation
Building an ETL and Machine Learning Pipeline for Udacity Data Scientist ND Project-2. Utilizing Plotly, Flask, Pandas, Scikit-learn, NLTK, SQLite, and more.

### Objective:
Analyzing disaster data from Figure Eight to construct a model for an API classifying messages sent during disaster events. The pipeline categorizes real messages for appropriate routing to disaster relief agencies. The project also includes a web app allowing emergency workers to input new messages and receive classification results across various categories, along with visualizations.

### File Descriptions
app    

| - template    
| |- master.html # main page of web app    
| |- go.html # classification result page of web app    
|- run.py # Flask file that runs app    


data    

|- disaster_categories.csv # data to process    
|- disaster_messages.csv # data to process    
|- process_data.py # data cleaning pipeline    
|- DisasterResponse.db # database to save clean data to     


models   

|- train_classifier.py # ML pipeline     
|- classifier.pkl # saved model using pickle     


README.md    

### Components
There are three components I completed for this project. 

#### 1. ETL Pipeline
`process_data.py`: cleans the data through the use of an ETL approach:

 - Loads data from two datasets categories and messages (EXTRACT)
 - Merges the two datasets (TRANSFORM)
 - Cleans the data(TRANSFORM)
 - Stores the cleaned data in SQLite database (LOAD)
 
If you're interested in understanding how the Pipeline is prepared, please refer to the Jupyter notebook named `ETL Pipeline Preparation`

#### 2. ML Pipeline
`train_classifier.py`: a machine learning pipeline

 - Loads data from the SQLite database (ResponseDisaster.db)
 - Splits the data into training and testing sets.
 - Processes text through a text-processing pipeline 
 - Creates a machine learning pipeline
 - Prints classification report
 - Saves the final model as a pickle file
 
If you're interested in understanding how the Pipeline is prepared, please refer to the Jupyter notebook named `ML Pipeline Preparation`

#### 3. Flask Web App
The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. The outputs are shown below:

![image](https://github.com/FahadAlAraik/Udacity-DataScienceND-Project-Two/assets/51764194/8779ad1c-ba44-4d6f-b936-bacfa0acae73)

![image](https://github.com/FahadAlAraik/Udacity-DataScienceND-Project-Two/assets/51764194/f8fe9ea7-7c44-48d0-a9b6-ed74a5fd4051)



### Instructions on How to Interact With the Project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
	`cd app`
    <br />
    `python run.py`

3. Go to http://0.0.0.0:3000/
