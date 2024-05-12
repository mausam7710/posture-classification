import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

datasets = [
    '53M_test.csv', '62M_test.csv', '69M_test.csv', '75M_test.csv', '77M_test.csv',
    '86M_test.csv', '93M_test.csv', '49F_test.csv', '51F_test.csv', '60F_test.csv',
    '72F_test.csv', '78F_test.csv', '83F_test.csv'
]
features = ['LC1', 'LC2', 'LC3', 'LC4', 'LC5', 'LC6', 'GAS_X', 'GAS_Y', 'GAS_Z']
target = 'Class'

models = {
    'SVM': joblib.load('svm.pkl'),
    'Random Forest': joblib.load('rf.pkl'),
    'Logistic Regression': joblib.load('mnr.pkl'),
    'CNN': load_model('cnn.h5'),
    'RNN': load_model('rnn.h5')
}

scalers = {
    'CNN': joblib.load('cnn_scaler.pkl'),
    'RNN': joblib.load('rnn_scaler.pkl')
}

def prepare_and_predict(model, X, y, model_name):
    if model_name in ['CNN', 'RNN']:

        scaler = scalers[model_name]
        X_scaled = scaler.transform(X)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        y_pred_probs = model.predict(X_scaled)
        y_pred = np.argmax(y_pred_probs, axis=1)
    else:
        y_pred_probs = model.predict_proba(X)
        y_pred = np.argmax(y_pred_probs, axis=1)
    
    precision = precision_score(np.argmax(y.values, axis=1), y_pred, average='weighted')
    recall = recall_score(np.argmax(y.values, axis=1), y_pred, average='weighted')
    
    return precision, recall, y_pred_probs

results = {}

for dataset in datasets:
    data = pd.read_csv(dataset)
    X = data[features]
    y = pd.get_dummies(data[target])
    results[dataset] = {}
    
    for model_name, model in models.items():
        precision, recall, y_pred_probs = prepare_and_predict(model, X, y, model_name)
        
        results[dataset][model_name] = {
            'Precision': precision,
            'Recall': recall
        }


result_df = pd.DataFrame({(dataset, model): metrics for dataset, models_metrics in results.items() for model, metrics in models_metrics.items()}).T

result_df[['Precision', 'Recall']] = result_df[['Precision', 'Recall']].astype(float)

result_df[['Precision', 'Recall']].round(2).to_csv('precision-recall.csv')