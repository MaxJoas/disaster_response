import json
import joblib
import numpy as np
import pandas as pd
import plotly
from flask import Flask, jsonify, render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

from disaster.models.train_classifier import tokenize

app = Flask(__name__, static_url_path='/static')

# load data
engine = create_engine('sqlite:///disaster/data/disaster.db')
df = pd.read_sql_table('mytable', engine)
plotting_helper = pd.read_csv('disaster/data/plotting_df.csv')

# load model
model = joblib.load("disaster/models/disaster_model.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    categories_df = df.drop(
        columns=['id', 'message', 'genre'])
    # define variable for plotting
    categories_counts = plotting_helper['categories_counts']
    categories_names = list(categories_df.columns)
    categories_names = [name.replace("_", " ") for name in categories_names]

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Category"
                },
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)'
            }
        },
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=plotting_helper['av_word_frequency']
                )
            ],

            'layout': {
                'title': 'Average Word Frequency per Category',
                'yaxis': {
                    'title': "Mean Word Frequency"
                },
                'xaxis': {
                    'title': "Message Category"
                },
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)'
            }
        }
    ]

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
