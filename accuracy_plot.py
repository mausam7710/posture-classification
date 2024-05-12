import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
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
    'MNR': joblib.load('mnr.pkl'),
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
        y_pred = np.argmax(model.predict(X_scaled), axis=1)
        accuracy = accuracy_score(np.argmax(y, axis=1), y_pred)
    else:
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
    return accuracy

results = {}

for dataset in datasets:
    data = pd.read_csv(dataset)
    X = data[features]
    y = data[target]
    results[dataset] = {}
    
    for model_name, model in models.items():
        if model_name in ['CNN', 'RNN']:

            y_encoded = pd.get_dummies(y).values
            acc = prepare_and_predict(model, X, y_encoded, model_name)
        else:
            acc = prepare_and_predict(model, X, y, model_name)
        
        results[dataset][model_name] = acc

result_df = pd.DataFrame(results).T

plt.figure(figsize=(16, 10))
sns.heatmap(result_df, annot=True, fmt=".2f", cmap='viridis', square=True, annot_kws={"size": 9})
plt.ylabel('Dataset')
plt.xlabel('Model')

plt.yticks(ticks=np.arange(len(datasets)) + 0.5, labels=[dataset[:-9] for dataset in datasets])

plt.xticks(ticks=np.arange(len(models)) + 0.5, labels=['SVM', 'RF', 'MNR', 'CNN', 'RNN'])

plt.savefig("accuracy_score.svg", format='svg')
plt.show()
