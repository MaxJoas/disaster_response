import pandas as pd
import numpy as np

from sqlalchemy import create_engine
from disaster.models.train_classifier import tokenize


# lambda function to get the token count of each message
def get_message_length(x): return len(tokenize(x))


# read in cleaned dataframe
engine = create_engine('sqlite:///disaster/data/disaster.db')
df = pd.read_sql_table('mytable', engine)

# get dataframe with only the categories
categories_df = df.drop(columns=['id', 'message', 'genre'])

# get count of each category
categories_counts = categories_df.sum()

# get nice names without _ for plotting
categories_names = list(categories_df.columns)
categories_names = [name.replace("_", " ") for name in categories_names]

# add column that indicates message length in words
df['message_length'] = df['message'].apply(get_message_length)

# Now I want to get the average message length of each category
# Therefore I replace the indicator 1 with the length of the actual message
helper = categories_df.copy()
for col in helper.columns:
    helper[col] = np.where(
        helper[col] == 1, df['message_length'], df[col])

# in order to get the average word frequency per category I sum the length of
# message. Attention! I only divide by the count of message that actually
# belong to the category. In prevent dividing by zero I add a small
# amount to the sum of message length
word_freq = np.sum(helper) / (np.sum(helper != 0) + 0.001)

# building and saving the dataframe to a csv file
plotting_helper_df = pd.DataFrame({'av_word_frequency': word_freq,
                                   'categories_counts': categories_counts})
plotting_helper_df.to_csv('disaster/data/plotting_df.csv')
