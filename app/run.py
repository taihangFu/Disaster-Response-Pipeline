import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

import operator
from text_preprocessing import preprocess

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract data needed for visuals
    # distributions of message category
    msg_cats_count = df[df.columns[4:]].sum() #count
    msg_cats_count = msg_cats_count.sort_values(ascending = False)  
                                                          
    msg_cats = list(msg_cats_count.index)  # category names                            
    
     # extract data needed for visuals
     # top 10 frequent words 
    
    '''
    1. CONCAT ALL MESSAGES ON DF TOGETHER AS A STRING
    2. CALL PRERPOCESS
    3. get the return frequency dict and plot
    '''
    contents = df['message'].values
    contents = ' '.join(contents.flatten().tolist())
    word_frequency=preprocess(contents=contents, punctuations=True, stopwords=True, lemmatization=True)
   
    sorted_word_frequency = sorted(word_frequency.items(),  #list of tuples sorted by dictionary value(freq count of each word)
                                         key=operator.itemgetter(1),# sorted by value
                                         reverse=True) 
    msg_words, msg_words_count = map(list, zip(*sorted_word_frequency)) #list of tuples -> list(s)     
    msg_words = msg_words[:10]#takes top-10 only
    msg_words_count = msg_words_count[:10] #percentage
        
    #print(word_frequency)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        #barchart of genre of messages
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
        #TODO: bar chart of classification labels
        {
            'data': [
                Bar(
                    x=msg_cats,
                    y=msg_cats_count
                )
            ],

            'layout': {
                'title': 'Distribution of Messages Category',
                'yaxis': {
                    'title': "Count",
                    'automargin':True
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -40,
                    'automargin':True
                }
            }
        
        },
        #TODO: bar chart of top-10 messages
        {
            'data': [
                 Bar(
                            x=msg_words,
                            y=msg_words_count
                        )
            ],

            'layout': {
                'title': 'Top 10 frequent words',
                'yaxis': {
                    'title': 'Occurrence',
                    'automargin': True
                },
                'xaxis': {
                    'title': 'Top 10 words',
                    'automargin': True
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