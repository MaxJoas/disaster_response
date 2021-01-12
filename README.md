## Disaster Response Pipeline Project
### Motivation
balblabalbal

### Instructions:

1. Install and Run the whole Pipeline:
`./main.sh`
1. Run the following commands in the project's root directory to run the steps separatly.

    - To run ETL pipeline that cleans data and stores in database
        `python3 disaster/data/process_data.py disaster/data/disaster_messages.csv disaster/data/disaster_categories.csv disaster/data/disaster.db`
     - To Prepate the Visualization
        `python3 disaster/data/prepare_plotting.py`
    - To run ML pipeline that trains classifier and saves
        `python3 disaster/models/train_classifier.py disaster/data/disaster.db disaster/models/disaster_model.pkl`
   
        

2. Run the following command in the app's directory to run your web app.
    `python3 disaster/app/run.py`

3. Go to http://0.0.0.0:3001/

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




