import sys
# import libraries
import sqlalchemy as sql
import pandas as pd
import pickle
import numpy as np

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def load_data(database_filepath):
    """
    Load the data from the SQLite database
    Input:
    database_filepath (str): path of the database file
       
    Returns:
    X (data frame): The messages
    y (data frame): The classifications of the messages
    categories_names (list): list of all categories
    """    
    
    tbname = 'dr_messages_tbl'
    
    # connect to the database
    conn = sql.create_engine('sqlite:///{}'.format(database_filepath)).connect()
    
    #read the data from a table into a data frame
    df = pd.read_sql_table(tbname, con=conn)  
    
    df.drop(['child_alone'], axis=1, inplace=True)
    
    df['related'] = df['related'].apply(lambda x: 1 if x == 2 else x)
    
    X = df.message
    
    y = df.drop(['id', 'message', 'original'], axis=1)
    
    categories_names = list(np.array(y.columns))
    
    return X, y, categories_names

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
    
    tokens = nltk.word_tokenize(text)

    #tokens = [word for word in tokens if not word in stopwords.words('english')]

    clean_tokens = []
    
    lemmatizer = nltk.WordNetLemmatizer()
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        # tokenize by sentences
        sentence_list = nltk.sent_tokenize(text)
        
        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            pos_tags = nltk.pos_tag(tokenize(sentence))

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

def build_model():
    '''
    Creates a pipeline object with a countvectorizer, tfidtransformation, starting verb extractor and a
    multioutputclassifier with a ada boost classifier estimator.
    
    Input:
    None:
       
    Returns:
    pipeline (pipeline): An initialized pipeline object

    '''
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            
            ('starting_verb', StartingVerbExtractor())
        ])),
        
        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    return pipeline

def optimize_model_params(model, X_train, Y_train):
    '''
    Optimize the model using grid search.
    
    Input:
    model: model object
    X_train (dataframe): Messages from training data set          
    Y_train (dataframe): Classified categories for each message in training data set 
       
    Returns:
    model: A model object with optimized parameters

    '''    
    
    parameters = {
            'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
            'features__text_pipeline__vect__max_df': (0.75, 1.0),
            'features__text_pipeline__vect__max_features': (None, 5000),
            'features__text_pipeline__tfidf__use_idf': (True, False)
            }
    
    cv = GridSearchCV(model, param_grid=parameters, scoring='f1_micro')
    
    cv.fit(X_train, Y_train)
    
    return cv
    
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the results of the model by predicting categories from testing data.
    
    Input:
    model (): model object
    X_test (dataframe): Messages for testing model           
    Y_test (dataframe): Classified categories for each message 
    category_names (list): List of categories 
       
    Returns:
    None.

    '''
    print ('Predicting on test data...')
    y_predicted_test = model.predict(X_test)
    
    print (classification_report(Y_test, y_predicted_test, target_names=category_names))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    print ('Model saved...')

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
        
        #print('Optimizing parameters of model...')
        #model = optimize_model_params(model, X_train, Y_train)
        
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