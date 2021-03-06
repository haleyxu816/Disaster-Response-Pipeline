import json
import plotly
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet','stopwords','averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import string
from nltk.stem.porter import PorterStemmer
import re


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine




app = Flask(__name__)



def get_wordnet_pos(treebank_tag):
    '''
    Function that map treebank tags to WordNet part of speech names
    '''
    
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV  
    else:
        return wordnet.NOUN


def tokenize(text):
    '''
    the tokenize function would process the text data
    '''
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word,get_wordnet_pos(pos)) for word,pos in nltk.pos_tag(tokens) if word not in stop_words]
    tokens =[PorterStemmer().stem(word) for word in tokens]
    return tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster message', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visual 1
    genre_counts_related = df[df['related']==1].groupby('genre').count()['message']
    genre_names_related = list(genre_counts_related.index)
    
    genre_counts_unrelated = df[df['related']==0].groupby('genre').count()['message']
    genre_names_unrelated = list( genre_counts_unrelated.index)
    
    # extract data needed for visual 2
    categories_count=df.iloc[:,5:].apply(sum)
    categories_name=list(df.columns[5:])
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names_related,
                    y=genre_counts_related,
                    name='related message'
                ),
                 Bar(
                    x=genre_names_unrelated,
                    y=genre_counts_unrelated,
                    name='unrelated message'
                )
                
            ],
                   
            'layout': {
                'barmode' : 'stack',
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            } 
         },
        { 'data': [
            Bar(
                x=categories_name,
                y=categories_count
            ) ],
         'layout': {
             'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
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
    classification_labels = model.predict(tokenize(query))[0]
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
