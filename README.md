## Disaster Response Pipeline Project
### Motivation
This project was part of my Udacity Data Science Nano Degree. The degree encouraged me to solve real world problems with Data Science.\
In this project I will analyse Tweets from people that suffered from a disaster (natural catastrophe, etc.). Disaster help organization need to provide the relevant aid. Therefore I implemented a ML Pipeline that uses Natural Language Processing to analyse the Tweets and predict a disaster category.
The ML models were trained with a prelabeled dataset from Udacity. To categorize new messages, I build a Flask Web App.

### Instructions:

1. Install and Run the whole Pipeline:
`./main.sh`
2. Run the following commands in the project's root directory to run the steps separately.

    - `pip install -e .'
    - To run ETL pipeline that cleans data and stores in database\
        `python3 disaster/data/process_data.py disaster/data/disaster_messages.csv disaster/data/disaster_categories.csv disaster/data/disaster.db`
     - To Prepare the Visualization\
        `python3 disaster/data/prepare_plotting.py`
    - To run ML pipeline that trains classifier and saves\
        `python3 disaster/models/train_classifier.py disaster/data/disaster.db disaster/models/disaster_model.pkl`



3. Run the following command in the app's directory to run your web app.
    `python3 disaster/app/run.py`

4. Go to http://0.0.0.0:3001/

### Requirements
The following libraries have been used:
*  Python 3.6.9
*  Pandas 1.1.2
*  Numpy 1.19.2
*  Matplotlib 3.3.3
*  Flask 1.1.2
*  joblib 0.16.0
*  json5 0.9.5
*  nltk 3.5
*  plotly 4.14.1
*  SQLAlchemy 1.3.22
*  scikit-learn 0.23.2
*  pickleshare==0.7.5

### File Structure
*  `setup.py` Used to install the package\
*  `main.sh` Wrapper script to run the whole pipeline\
*  The `disaster` directory contains all files and folder for the package\
*  The `disaster/data` directory contains the input data file `disaster_messages.csv` and `disaster_categories.csv` as well as a script to clean `process_data.py` the data and prepare the data for plotting `prepare_plotting.py`. The cleaned data will be also stored here as `disaster.db`\
* The `disaster/models` directory contains `train_model.py` to train the model as well as the trained model as `disaster_model.pkl`\
* The `disaster/app` directory contains the starting point for the Flask app `run.py`. Inside this directory are the `templates` and `static` directories containing `master.html` and `go.html` for the Web App and `styles.css` for the custom styling of the Web App.
* The `results` folder contains metrics for the prediction.

### Acknowledgements
Udacity provided a Basic Structure for the Flask Web App. However I made a lot of changes to have a more state of the art design.
The Data was provided by Udacity from their Partner FigureEight.

