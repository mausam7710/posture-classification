import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv('train.csv')
X = data[['LC1', 'LC2', 'LC3', 'LC4', 'LC5', 'LC6', 'GAS_X', 'GAS_Y', 'GAS_Z']].values
y = data['Class'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

y = y - 1 
num_classes = 5 
y_encoded = to_categorical(y, num_classes=num_classes)
X = X.reshape(X.shape[0], X.shape[1], 1) 

model = Sequential()
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(9, 1)))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y_encoded, epochs=10, batch_size=128) 

model.save('cnn.h5')
joblib.dump(scaler, 'cnn_scaler.pkl') 
