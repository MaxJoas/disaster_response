import pandas as pd
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import sent_tokenize
from disaster.models.train_classifier import *


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    The StartingVerbExtractor is a transformer that ca be used in a sklearn
    Pipeline. The transformer transforms a text message to a boolean, which
    indicated whether the message starts with a Verb

    Args:
        None

    Methods:
        fit
        transform
    """

    def starting_verb(self, text):
        """ Determines whether a message starts with a verb

        Args:
            text (str): text message to analyze
        Returns:
            boolean

        """
        # tokenize by sentences
        sentence_list = sent_tokenize(text)

        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) == 0:
                return False

            # index pos_tags to get the first word and part of speech tag
            first_word, first_tag = pos_tags[0][0], pos_tags[0][1]

            # return true if the first word is an appropriate verb or RT for retweet
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True

            return False

    def fit(self, x, y=None):
        """Fits the custom transformer StartingVerbExtractor()

        Args:
            x (pd.Series): pandas Series of text messages
            y (None): None
        Returns:
            self

        """
        return self

    def transform(self, X):
        """ transforms a Vector of text messages to a boolean that indicates
        if the text message starts with a verb

        Args:
            X (pd.Series): pandas Series of text messages

        Returns:
            res (pd.DataFrame): pandas dataframe of booleans

        """
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(lambda x: self.starting_verb(x)).values
        res = pd.DataFrame(X_tagged)
        # in case a transformation fails and would result in na values, I fill
        # the replcase the na values with a zero in order to further process
        # the data in a ML model
        res.fillna(0, inplace=True)
        return res
