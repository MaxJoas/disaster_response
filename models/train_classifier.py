import re
import sys

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

nltk.download('stopwords')


def load_data(database_filepath, table_name):
    """ loads data from a sql db to a pandas dataframe, and returns two
    dataframes, one for the explanatory variables, one for the prediction
    target as well a list of names for the prediction targets

    Args:
        database_filepath (str): path to sql database
        table_name (str): name of the sql table
    Returns:
        X (pd.DataFrame): explanatory variables
        y (pd.DataFrame): prediction target categories
        category_names: (list(str)): names of categories

    """
    # create db connection
    path = 'sqlite:///' + database_filepath
    engine = create_engine(path)
    df = pd.read_sql_table(table_name=table_name, con=engine)
    df.columns
    X = df['message']
    y = df.drop(columns=['id', 'message', 'genre'])
    category_names = list(y.columns)

    return X, y, category_names


def tokenize(text):
    """ tokenizes and lemmatizes a text in order to further process the text
    with CountVectorizer to use in a ML model later on.

    Args:
        text (str): a text message

    Return:
        clean_tokens (list(str)): list of processed words

        """
    # instantiate Lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text = make a list of where each word is a separate list entry
    tokens = word_tokenize(text)

    # lemmatize (=get word stem) and remove stop words ('a', 'and', etc.)
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens
                    if word not in stop_words]

    return clean_tokens


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 4:
        database_filepath, model_filepath, table_name = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

