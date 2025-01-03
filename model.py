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

mlflow.set_experiment('water_potability_prediction_gbt')
mlflow.set_tracking_uri('http://localhost:5000')

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

with mlflow.start_run():
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('min_samples_split', min_samples_split) 

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

    plt.figure(figsize=(5, 5))
    sns.heatmap(matrix, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    plt.savefig('confusion_matrix.png')

    mlflow.log_artifact('confusion_matrix.png')

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('f1', f1)
    
    mlflow.sklearn.log_model(model, 'GradientBoostingClassifier')

    mlflow.log_artifact(__file__)
    mlflow.set_tag('mlflow.author', 'Digvijay')
    mlflow.set_tag('mlflow.source.type', 'script')
    mlflow.set_tag('mlflow.model', 'gration_boosting_classifier')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')    
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
