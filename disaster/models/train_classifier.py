from nltk.tokenize import sent_tokenize
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
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sqlalchemy import create_engine
from disaster.models.starting_word import *
from collections import defaultdict


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
    # path = 'sqlite:///' + './disaster/data/test.db'

    path = 'sqlite:///' + database_filepath
    engine = create_engine(path)
    # read sql into pandas dataframe
    df = pd.read_sql_table(table_name='mytable', con=engine)

    # get explanatory and target variables as pandas dataframe / Series
    X = df['message']
    y = df.drop(columns=['id', 'message', 'genre'])
    # X = X[0:50]
    # y = y.iloc[0:50, :]
    category_names = list(y.columns)

    return X, y, category_names


def tokenize(text):
    """ tokenizes and lemmatizes a text in orde r to further process the text
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

            # ('starting_verb', StartingVerbExtractor()),
            # ('pos_frequency', PosFrequency())
        ])),

        # ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
        ('clf', MultiOutputClassifier(MLPClassifier(random_state=42,
                                                    max_iter=500)))

    ])

    # define tuning parametes for grid search, syntax can be explained as
    # follows: chained names of step in pipeline separated with two underscores
    # followed by actual parameter name of the step

    parameters = {
        'clf__estimator__hidden_layer_sizes': [(50, 50, 50), (100,)],
        'clf__estimator__activation': ['tanh', 'relu'],
        'clf__estimator__alpha': [0.0001, 0.05],
        'clf__estimator__learning_rate': ['constant', 'adaptive'],
    }
    # instantiating Grid Search
    cv = GridSearchCV(pipeline, param_grid=parameters,
                      cv=3, verbose=1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """evaluates a supervised classification model with recall, precision and
    f1_score.

    Args:
        model (obj): fitted supervised classification model
        X_test (pd.Series): feature matrix of test data set
        Y_test (pd.DataFrame): target categories of test data set
    Returns:
        None

    """
    best_model = model.best_estimator_  # get best model
    # print best estimator
    print('The best model has this parameter: {}'.format(model.best_params_))
    y_preds = best_model.predict(X_test)  # predicts message categories
    pd.DataFrame(y_preds).to_csv('predictions.csv')
    Y_test.to_csv('true_categories.csv')

    # In order to evaluate the model I break the multiclass problem down
    # into a single class problem by comparing each category separately
    y_preds_t = y_preds.T  # transform to better compare each category
    Y_test_array = Y_test.to_numpy()
    Y_test_array = Y_test_array.T

    # I will save the recall, py arrprecision and f1_score for each category in
    # dictionary, whereby the metrics will be the keys
    helper = defaultdict(list)
    for i in range(len(category_names)):
        recall = recall_score(y_true=Y_test_array[i],
                              y_pred=y_preds_t[i],
                              average='micro')
        helper['recall'].append(recall)

        precision = precision_score(y_true=Y_test_array[i],
                                    y_pred=y_preds_t[i],
                                    average='micro')
        helper['precision'].append(precision)

        f1 = f1_score(y_true=Y_test_array[i],
                      y_pred=y_preds_t[i],
                      average='micro')
        helper['f1'].append(f1)

        print('The category {} has a recall of: {}, a precision of: {} and \
              a f1_score of {}'.format(category_names[i], recall, precision, f1))

    res = pd.DataFrame(helper)
    res.index = category_names
    res.to_csv(path_or_buf='metrics.cav')
    print(res)


def save_model(model, model_filepath):
    """ saves fitted model to a pickle file

    Args:
        model (object): fitted ML model
        model_filepath (str): path where the model will be stored

    Returns:
        None

    """
    with open(file=model_filepath, mode='wb') as f:
        pickle.dump(file=f, obj=model)


def main():
    """ Reads in the databasel filepath and the filepath where the final
    model will be stored via commandline arguments. Calls functions for
    loading the data and training and evaluating the ML model

    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42)

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
