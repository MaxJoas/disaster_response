
echo "Cleaning Workspace"
mkdir old
mv *.csv ./old
mv disaster/models/*.pkl ./old

mv disaster/data/disaster.db ./old
echo "Install module"
pip3 install -e .

echo "Cleaning Data"
python3 disaster/data/process_data.py disaster/data/disaster_messages.csv disaster/data/disaster_categories.csv disaster/data/disaster.db

echo "Preparing Data For Plots"
python3 disaster/data/prepare_plotting.py

echo "Training Model - This can takes 12h with 8 threads and 16 GB RAM"
python3 disaster/models/train_classifier.py disaster/data/disaster.db disaster/models/disaster_model.pkl
printf "=====================================================================\n"
echo "prediction results in results directory"
echo "Model in ./disaster/models/"
mkdir results
mv *.csv ./results

printf "=====================================================================\n"
echo "starting web app"
python3 disaster/app/run.py


