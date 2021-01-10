import pandas as pd
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import sent_tokenize
from disaster.models.train_classifier import tokenize


class PosFrequency(BaseEstimator, TransformerMixin):

    def __init__(self, pos_tag='NN'):
        self.pos_tag = pos_tag

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_counted = pd.Series(X).apply(lambda x: self.pos_frequency(x)).values
        res = pd.DataFrame(X_counted)
        res.fillna(0, inplace=True)
        return res

    def pos_frequency(self, text):
        counter = 0
        tokenized = tokenize(text)
        pos_tags = nltk.pos_tag(tokenized)
        for tag in pos_tags:
            if tag[1] == self.pos_tag:
                counter += 1
                return counter / len(tokenized)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
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
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(lambda x: self.starting_verb(x)).values
        res = pd.DataFrame(X_tagged)
        res.fillna(0, inplace=True)
        return res
# starting = StartingVerbExtractor()
# starting.fit(X)
# df_trans = starting.transform(X)
# count = PosFrequency()
# count.fit(X)
# df_pos = count.transform(X)
