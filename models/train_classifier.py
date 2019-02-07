import sys
from sqlalchemy import create_engine
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet','stopwords','averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import string
from nltk.stem.porter import PorterStemmer
import re



from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report


import pickle

from workspace_utils import active_session


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster message',con=engine)
    X = df.message.values
    Y = df.iloc[:,4:].values
    category_names=df.columns[4:]
    return X,Y,category_names

from nltk.corpus import stopwords
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV  
    else:
        return wordnet.NOUN

# normalize case and remove punctuation，numbers
def tokenize(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize and remove stop words    
    tokens = [lemmatizer.lemmatize(word,get_wordnet_pos(pos)) for word,pos in nltk.pos_tag(tokens) if word not in stop_words]
    # Reduce words to their stems
    tokens =[PorterStemmer().stem(word) for word in tokens]
    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('multioutput',MultiOutputClassifier(LinearSVC(class_weight='balanced'),n_jobs=-1))
    ])
    
    parameters = {
       'vect__ngram_range': ((1, 1), (1, 2)),
       'vect__max_features': (None,5000,10000),
       'tfidf__use_idf': (True,False),
       'multioutput__estimator__C': [100,1000,10000]
   }
    
    with active_session():
        cv = GridSearchCV(pipeline, param_grid=parameters,cv=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i in range(35):
        print ('classification report of {}:'.format(category_names[i]))
        print(classification_report(Y_test[:,i], Y_pred[:,i]))
    

def save_model(model, model_filepath):
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