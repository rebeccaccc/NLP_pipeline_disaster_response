import json
import plotly
import re
import numpy as np
import pandas as pd
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
from sqlalchemy import create_engine

# model evaluaion
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

# model build
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Classifiers
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, TransformerMixin


app = Flask(__name__)

# function to indicate if the first word is verb
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''function to indicate if the first word is verb '''
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if(len(pos_tags) > 0):
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
                else:
                    return False
            else: 
                return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


class StartingNounExtractor(BaseEstimator, TransformerMixin):
    '''function to indicate if the first word is noun '''
    def starting_noun(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if(len(pos_tags) > 0):
                first_word, first_tag = pos_tags[0]
                if first_tag in ['NN', 'NNS','NNPS','NNP']:
                    return True
                else:
                    return False
            else: 
                return False


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_noun)
        return pd.DataFrame(X_tagged)

def tokenize(text):
    
    ''' This function removes url, remove punctuations and symbols, tokenize words and removes stop words.'''
    
    # remove urls if any
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # remove all punctuations and symbols 
    text = re.sub(r'[^a-zA-Z0-9]', " ", text)
    # tokenize words
    tokens = word_tokenize(text)
    # lemmatize words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    # remove stop words 
    clean_tokens = [t for t in clean_tokens if t not in nltk.corpus.stopwords.words("english")]
    
    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/response_online.db')
df = pd.read_sql_table('response', engine)

# load model
#model = joblib.load("../models/classifier.pkl")
model = pickle.load(open("../models/classifier0405.pkl", 'rb'))

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visual
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    corr_mtrx = df[['request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']].corr()['direct_report'].drop('direct_report')
    corr_df = pd.DataFrame({'Message_Category': corr_mtrx.index, 'Correlation' : corr_mtrx}).reset_index().drop('index', axis = 1)
    
    msg_cat = df[['request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']].mean()
    msg_cat = pd.DataFrame({'Message_Category': msg_cat.index, 'Pct' : msg_cat}).reset_index().drop('index', axis = 1)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }, 
         {
            'data': [
                Bar(
                    y=corr_df['Correlation'],
                    x=corr_df['Message_Category']
                )
            ],

            'layout': {
                'title': 'Correlation between Direct Report and Message Category',
                'yaxis': {
                    'title': "Correlation with Direct_report"
                }
            }
        }, 
        {
            'data': [
                Bar(
                    y=msg_cat['Pct'],
                    x=msg_cat['Message_Category']
                )
            ],

            'layout': {
                'title': 'Proportion of Messages in Each Message Category',
                'yaxis': {
                    'title': "Proportion of Messages"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()