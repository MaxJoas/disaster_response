import sys
import pandas as pd

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Reads two csv files and concatenates them to a pandas dataframe

    Args:
        messages_filepath (str): path to the messages csv file
        categories_filepath (str): path to the categories_filepath"

    Returns:
        df (pd.DataFrame): concatinated dataframe

    """
    # read in files
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)

    # merge the to dataframes based on the id
    # note I assume the id to have no missing values
    df = df_messages.merge(df_categories, how='outer', on='id')

    return df


def clean_data(df):
    """Encodes message categories, removes duplicates and deals with na values

    Args:
        df(pd.DataFrame): a pandas dataframe

    Returns:
        df (pd.DataFrame): the cleaned dataframe

    """
    # put each category in the categories (list) into its own col and rename
    cat_col = df['categories'].str.split(';', expand=True)
    new_col_names = list(df['categories'][0].split(';'))
    new_col_names = [x[:len(x) - 2] for x in new_col_names]
    cat_col.columns = new_col_names

    # extract the binary value that indicated whether a message belongs to
    # a category so 'request -1' becomes 1
    def extract_binary(x): return int(x[len(x) - 1])  # lambda function
    for col in new_col_names:
        cat_col[col] = cat_col[col].apply(extract_binary)

    # concatenate the new cols to the df and drop the old categories col
    df = pd.concat([df, cat_col], axis=1)
    df.drop(columns=['categories'], inplace=True)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # We cannot allow any na values in the categories cols, so we drop every
    # observation that has na values in any of the categories
    # Because we use the message to make predictions do the same here
    new_col_names.append('message')
    before_observations, before_cols = df.shape
    df.dropna(subset=new_col_names, inplace=True)
    after_observations = df.shape[0]
    dif = before_observations - after_observations
    print('Dropping {} rows, due to missing messages / categories'.format(dif))

    # The other columns are not super relevant for our model, so out of
    # simplicity we drop the remaining columns that have missing values
    na_cols = list(df.columns[df.isna().mean() > 0])  # get col names with na
    df.drop(columns=na_cols, inplace=True)
    after_columns = df.shape[1]
    dif_cols = before_cols - after_columns
    print('Dropping {} cols, due to missing values'.format(dif_cols))

    return df


def save_data(df, database_filename):
    """Saves the cleaned dataframe to a SQL database

    Args:
        df (pd.DataFrame): cleaned pandas dataframe
        database_filename (str): name of database
    Returns:
        None

    """
    path = 'sqlite:///' + str(database_filename)
    engine = create_engine(path)
    df.to_sql(database_filename, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
