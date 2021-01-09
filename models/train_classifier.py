import pickle
import re
import sys

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
from starting_word import *

sys.path.append('.')


# from starting_word import StartingVerbExtractor

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')


def load_data(database_filepath):
    """ loads data from a sql db to a pandas dataframe, and returns two
    dataframes, one for the explanatory variables, one for the prediction
    target as well a list of names for the prediction targets

    Args:
        database_filepath (str): path to sql database
    Returns:
        X (pd.DataFrame): explanatory variables
        y (pd.DataFrame): prediction target categories
        category_names: (list(str)): names of categories

    """
    # create db connection
    path = 'sqlite:///' + database_filepath
    engine = create_engine(path)
    # read sql into pandas dataframe
    df = pd.read_sql_table(table_name='mytable', con=engine)

    # get explanatory and target variables as pandas dataframe / Series
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
    # instantiate Lemmatizer and stopwords and regex pattern
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|\
        [!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # normalize case and remove punctuation
    detected_urls = re.findall(url_regex, text)
    # find urls in the messages and replace them with urlplaceholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # clean message from special chars and normalize (make everything lowercase)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text = make a list of where each word is a separate list entry
    tokens = word_tokenize(text)

    # lemmatize (=get word stem) and remove stop words ('a', 'and', etc.)
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens
                    if word not in stop_words]

    return clean_tokens


def build_model():
    """Builds a Pipeline that performs feature engineering with
    Count Vectorization and Tf-idf Transformation and tunes a supervised
    multiclass classification model with a cv Grid Search

    Args:
        None
    Returns:
        model (obj): tuned classification model

    """

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            # ('starting_verb', StartingVerbExtractor())
        ])),

        # ('clf', RandomForestClassifier())
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=4))
    ])

    # define tuning parametes for grid search, syntax can be explained as
    # follows: chained names of step in pipeline separated with two underscores
    # followed by actual parameter name of the step
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
            {'text_pipeline': 0.5, 'starting_verb': 1},
            {'text_pipeline': 0.8, 'starting_verb': 1},
        )
    }
    # instantiating Grid Search
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """evaluates a supervised classification model with a confusion matrix

    Args:
        model (obj): fitted supervised classification model
        X_test (pd.Series): feature matrix of test data set
        Y_test (pd.DataFrame): target categories of test data set
    Returns:
        None

    """
    best_model = model.best_estimator_  # get best model
    y_preds = best_model.predict(X_test)  # predicts message categories
    cm = confusion_matrix(y_pred=y_preds, y_true=Y_test, labels=category_names)
    # print confusion matrix
    print(cm)


def save_model(model, model_filepath):
    with open(file=model_filepath, mode='wb') as f:
        pickle.dump(file=f, obj=model)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
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
# for i in X:
#     try:
#         url = re.findall(i, url_regex)
#         if len(url) > 0:
#             print(i)
#     except:
#         text = i
#         print(i)
#         break

