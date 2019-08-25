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

nltk.download('stopwords')

def load_data(database_filepath):
    '''
    Load data from database and save to pandas dataframe

    Args:
        database_filepath (str): filepath where the database(created by data/process_data.py) is located

    Returns:
        X(pandas.DataFrame)
        Y(numpy.array)
        category_names (pandas.DataFrame.columns): the prediction labels name for Y
    '''
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine) #table name same as db name
    X = df['message']
    Y = df.iloc[:, 4:]

    category_names = Y.columns
    Y = Y.values#convert df back to numpy array for convenience used of sklearn prediction, calssification report
    
    return  X, Y, category_names 


def tokenize(text):
    '''
    Tokenize the given text by normalization(to lower cases, puntuation removeal), tokenization, stopwords removal,  lemmatization

    Args:
        text (str): filepath where the database(created by data/process_data.py) is located

    Returns:
        words: preprocessed tokenized text
    '''
    
    #normalize
    ##lower case
    text = text.lower() 
    ##remove puntuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    #tokenize text
    words = word_tokenize(text)
    
    #remove stopwords
    cachestopwords = set(stopwords.words("english")) #to speed up
    words = [w for w in words if w not in cachestopwords]
    
    #lemmatization
    words = list(set([WordNetLemmatizer().lemmatize(w) for w in words]))
    
    return words

def build_model():
     '''
     create a MultiOutputClassifier model by using pipelines and GridSearchCV for later trained

    Returns:
        cv: a  built models by GridSearchCV, used for training in the next stage
    '''
        
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1))), #TODO: test n_jobs=-1 efficiency
])
    
    parameters = {
    'clf__estimator__n_estimators': [50, 100]
}
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names): 
    '''
     make prediction on trained model then print out performance evaluation results
     
     Args:
        model (sklearn.multioutput.MultiOutputClassifier):
        X_test (pandas.DataFrame)
        Y_test (numpy.array)  
        category_names (pandas.DataFrame.columns): the prediction labels name for Y
    '''
    
    Y_pred = model.predict(X_test)
  
    for i in range(Y_test.shape[1]):
        print(category_names[i])
        print(classification_report(Y_test[:,i], Y_pred[:,i]))
        print()

def save_model(model, model_filepath):
    '''
    save model(pickle format) in given model_filepath
    
    Args:
       model (sklearn.multioutput.MultiOutputClassifier)
       model_filepath (str)
    '''
    
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
    main() 