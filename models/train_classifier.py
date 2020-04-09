#### import packages ####
import sys
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# data wrangling packages #
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# model evaluaion
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

# model build
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin



# Classifiers
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier


def load_data(database_filepath):
    ''' load data from a sql databased, cleans data and 
    removes message categories that has 0 variance.'''
    engine = create_engine(('sqlite:///'+ str(database_filepath)))
    df = pd.read_sql_table('response', engine)
    X = df.message.values
    Y = df.iloc[:,4:]
    # Remove columns with no variability
    Y = Y.loc[:,Y.std()!=0]
    
    catname = Y.columns
    
    return X , Y, catname


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

# function to indicate if the first word is verb
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    ''' Building a new text transformer to see if the first word is a verb.'''
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


# function to indicate if the first word is noun
class StartingNounExtractor(BaseEstimator, TransformerMixin):
    ''' Building a new text transformer to see if the first word is a noun.'''
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

def build_model():
    ''' build a pipeline to train, tune hyperparameter and test model'''
    
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('starting_verb', StartingVerbExtractor()), 
        ('starting_noun', StartingNounExtractor()),
        ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, ngram_range = (1,2))),
                ('tfidf', TfidfTransformer())
            ]))  
        ])),
    ('clf',MultiOutputClassifier(GradientBoostingClassifier()))
])
    
    # grid search 
    parameters = {
        'clf__estimator__learning_rate': [0.05, 0.1],
        'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__min_samples_split': [2, 4]
    }
    grid_srch = GridSearchCV(pipeline, param_grid = parameters)
    
    return grid_srch


def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluate model performance ysing f1_score, precision, recall and accuracy '''
    #predict outcomes based on the trained model with the best hyper-parameters
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns = category_names)
    
    # set bins to store metrics for model evaluation
    f1_sco = []
    precision = []
    recall = []
    accuracy = []
    for cat in Y_test : 
        acc = accuracy_score(Y_test[cat],Y_pred[cat])
        # the number of class is more than 2, the f1, precision and recall are not applicable. 
        if (Y_test[cat].nunique())>2:
            f1 = np.nan
            prcsn = np.nan
            rcl = np.nan
        
        else: 
            f1 = f1_score(Y_test[cat],Y_pred[cat])
            prcsn = precision_score(Y_test[cat],Y_pred[cat])
            rcl = recall_score(Y_test[cat],Y_pred[cat])
        
        f1_sco.append(f1)
        precision.append(prcsn)
        recall.append(rcl)
        accuracy.append(acc)
    
    
    rslt = pd.DataFrame({'Category' : category_names,
                         'f1_score' : f1_sco, 
                         'precision' : precision, 
                         'recall': recall, 
                         'accuracy' : accuracy})
    return rslt


def save_model(model, model_filepath):
    ''' save model into a pickel file'''
    saved_model = pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    ''' A main function to excute chained functions.'''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        # remove record with no words after tokenizing
        temp_len = []
        for t in X:
            temp_len.append(len(tokenize(t)))
        X = X[pd.Series(temp_len) != 0]
        Y = Y.loc[np.array(pd.Series(temp_len) != 0)]
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()