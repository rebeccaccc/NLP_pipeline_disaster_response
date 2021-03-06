# NLP Pipeline for Disaster Response

## Overview 
A natural language processing(NLP) pipeline that cleans, pre-processes and models published raw messages(e.g. twitter feeds, SMS, etc) in the event of natural disasters.

## Programming Language
<br> Python -- data cleaning; text pre-processing; building a NLP machine learning model pipeline to train models, tune hyperparameters and test model. 
<br>html -- Showcase restuls

## Packges and App
<br>General: numpy, pandas, sys, pickle
<br>Data cleaning: sqlalchemy, re 
<br>text pre-processing: nltk
<br>NLP machine learning modeling: sklearn

## File Structure of the Repository

### 1. data
#### process_data.py
The syntax for a ETL pipeline that cleans, wrangles, and saves data.
#### disaster_messages.csv
A dataset with messages published during natural disaster events
#### disaster_categories.csv
A dataset with message category information 
#### response_online.db
A SQL database storing the cleaned dataset.
### 2. models

#### train_classifier.py
The syntax of a NLP machine learing pipleline that pre-processes messages, builds NLP pipline, trains model, tunes hyperparameters, evaluates model performance, tests model and saves final model.
#### classifier0409.pkl
The saved trained model.
#### eval_result.csv
The model performance evaluation results. 

### 3. app
#### run.py
Code to showcase results on the FLASK web App. 
#### template
A folder has two html files: master.html and go.html. These files configure the result showcase dashboard on FLASK.

## Instruction
1. Run the following commands in the Terminal or a CMD window under the project's root directory to set up your database and model. Note that it will take about 10 hrs to train the NLP Machine Learning model. To view my results, skip step 1. 

    - To run ETL pipeline
        <br>`python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/response_online.db`
    - To run ML pipeline 
        <br>`python3 models/train_classifier.py data/response_online.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python3 run.py`

3. Now, open another Terminal window. Type `env|grep WORK`. You'll see output contains SPACEID and SPACEDOMAIN. 
<br> In a new web browser window, type in the following: `https://SPACEID-3001.SPACEDOMAIN`. Subsitute the corresponding elements in the web address with the SPACEID and SPACEDOMAIN in the terminal window. Then, press Enter.



