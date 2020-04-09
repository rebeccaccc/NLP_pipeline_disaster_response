# NLP Pipeline for Disaster Response

## Overview 
A natural language processing(NLP) pipeline that cleans, pre-processes and models raw published messages (e.g. twitter feeds, SMS, etc) when natural disasters happen.s

## Programming Language
<br> Python -- Data cleaning; text pre-processing; building a NLP machine learning model pipeline to train models, tune hyperparameters and test model. 
<br>html -- Showcase restuls

## Packges and App
<br>General: numpy, pandas, sys, pickle
<br>Data cleaning: sqlalchemy, re 
<br>text pre-processing: nltk
<br>NLP Machine Learning Modeling: sklearn

## File Structure of the Repository

### 1. data
#### process_data.py
The syntax for data cleaning, wrangling and saving cleaned data.
#### disaster_messages.csv
A dataset with messages published during natural disaster events
#### disaster_categories.csv
A dataset with message category information 
#### response_online.db
A SQL database storing the cleaned dataset.
### 2. models

#### train_classifier.py
The syntax to pre-process messages, build NLP pipline, train model, tune hyperparameters, evaluate model performance, test model and save final model.
#### classifier.0405.pkl
The saved trained model.

### 3. app

## Instruction


