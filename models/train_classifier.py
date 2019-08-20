import sys

# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine 

import pickle

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.multioutput import MultiOutputClassifier

import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine) #table name same as db name
    X = df['message']
    Y = df.iloc[:, 4:]

    category_names = Y.columns
    Y = Y.values#convert df back to numpy array for convenience used of sklearn prediction, calssification report
    
    return  X, Y, category_names 

def tokenize(text):
    #normalize  
    text = text.lower()   
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)  # remove puntuation 
    #tokenize text
    words = word_tokenize(text)
    #lemmatize
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in text]
    return lemmed


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1))), #TODO: test n_jobs=-1 efficiency
])
 
    return pipeline
 
def evaluate_model(model, X_test, Y_test, category_names): 
    Y_pred = model.predict(X_test)
  
    for i in range(Y_test.shape[1]):
        print(category_names[i])
        print(classification_report(Y_test[:,i], Y_pred[:,i]))
        print()

def save_model(model, model_filepath):
    #export pickle
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
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
    main()giot 