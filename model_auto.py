import pandas as pd 
import numpy as np
import pickle
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn


import dagshub
dagshub.init(repo_owner='Digvijay25', repo_name='mlflow_exp_dagshub', mlflow=True)

mlflow.set_experiment('water_potability_prediction_gbt')
mlflow.set_tracking_uri('https://dagshub.com/Digvijay25/mlflow_exp_dagshub.mlflow')

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/Digvijay25/Datasets/refs/heads/main/water_potability.csv')

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

def fill_missing_values_median(data):
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            data[column] = data[column].fillna(data[column].median())
    return data

train_processed_data = fill_missing_values_median(train_data)
test_processed_data = fill_missing_values_median(test_data)

X_train = train_processed_data.drop('Potability', axis=1)
y_train = train_processed_data['Potability']

n_estimators = 2000
max_depth = 20
min_samples_split = 20

mlflow.autolog()

with mlflow.start_run():

    clf = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    pickle.dump(clf, open('model.pkl', 'wb'))

    X_test = test_processed_data.drop('Potability', axis=1)
    y_test = test_processed_data['Potability']


    model = pickle.load(open('model.pkl', 'rb'))

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)   
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)



    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')    
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
