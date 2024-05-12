import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv('train.csv')

X = data[['LC1', 'LC2', 'LC3', 'LC4', 'LC5', 'LC6', 'GAS_X', 'GAS_Y', 'GAS_Z']]
y = data['Class']

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_classifier = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

best_rf = grid_search.best_estimator_

joblib.dump(best_rf, 'rf.pkl')
