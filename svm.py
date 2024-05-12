import pandas as pd
from sklearn.svm import SVC
import joblib

data = pd.read_csv('train.csv')

X = data[['LC1', 'LC2', 'LC3', 'LC4', 'LC5', 'LC6', 'GAS_X', 'GAS_Y', 'GAS_Z']]
y = data['Class']

svm_classifier = SVC(kernel='rbf', probability=True)

svm_classifier.fit(X, y)

y_pred = svm_classifier.predict(X)

joblib.dump(svm_classifier, 'svm.pkl') 
