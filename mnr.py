import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
import joblib

data = pd.read_csv('train.csv')  

print(data.isnull().sum())

X = data[['LC1', 'LC2', 'LC3', 'LC4', 'LC5', 'LC6', 'GAS_X', 'GAS_Y', 'GAS_Z']]
y = data['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

logreg_model = LogisticRegressionCV(Cs=np.logspace(-2, 2, 5), penalty='l2', solver='saga', max_iter=2000)
logreg_model.fit(X_scaled, y)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegressionCV(Cs=np.logspace(-2, 2, 5), penalty='l2', solver='saga', max_iter=2000))
])
pipeline.fit(X, y)

joblib.dump(pipeline, 'mnr.pkl') 
