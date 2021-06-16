import json
import plotly
import pandas as pd
import re
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from sklearn.base import TransformerMixin, BaseEstimator

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    Normalize and tokenize a piece of text
    Input:
    text (str): text from all messages
       
    Returns:
    clean_tokens (list): list of words into numbers of same meaning
    """    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")  
        
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    text = re.sub(r'[0-9]', " ", text)
    
    tokens = word_tokenize(text)

    #tokens = [word for word in tokens if not word in stopwords.words('english')]

    clean_tokens = []
    
    lemmatizer = WordNetLemmatizer()
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        # tokenize by sentences
        sentence_list = sent_tokenize(text)
        
        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            pos_tags = pos_tag(tokenize(sentence))

            # index pos_tags to get the first word and part of speech tag
            
            if not len(pos_tags): 
                return False
            first_word, first_tag = pos_tags[0]
            
            # return true if the first word is an appropriate verb or RT for retweet
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True

        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

# load data
engine = create_engine('sqlite:///data/DisasterResponseDB.db')
df = pd.read_sql_table('dr_messages_tbl', engine)

# load model
model = joblib.load("models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    most_common_sum = df.drop(['id', 'related', 'message'], axis=1).sum().sort_values(ascending=False).nlargest()
    most_common_cols = list(most_common_sum.index)
    
    least_common_sum = df.drop(['id', 'related', 'message'], axis=1).sum().sort_values(ascending=False).nsmallest()
    least_common_cols = list(least_common_sum.index)    
    
    percent_cols_pie = df.drop(['id', 'related', 'message'], axis=1).sum().sort_values(ascending=False)/len(df)
    all_cols_pie = list(df.drop(['id', 'related', 'message'], axis=1).columns)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        #Graph 1: Most common categories in training set
        {
            'data': [
                Bar(
                    x=most_common_cols,
                    y=most_common_sum
                )
            ],

            'layout': {
                'title': 'Most common categories in training set',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "All categories"
                }
            }
        },
        #Graph 2: Least common categories in training set
        {
            'data': [
                Bar(
                    x=least_common_cols,
                    y=least_common_sum
                )
            ],

            'layout': {
                'title': 'Least common categories in training set',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "All categories"
                }
            }
        },
        #Graph 3: Pie chart showing distribution of training data set
        {
            'data': [
                Pie(
                    labels=all_cols_pie,
                    values=percent_cols_pie
                )
            ],

            'layout': {
                'title': 'Pie chart showing distribution of training data set',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "All categories"
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