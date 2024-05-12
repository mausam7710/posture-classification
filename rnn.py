import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('train.csv')

X = df[['LC1', 'LC2', 'LC3', 'LC4', 'LC5', 'LC6','GAS_X', 'GAS_Y', 'GAS_Z']].values
y = df['Class'].values

y = y - 1 
num_classes = 5 
y_encoded = to_categorical(y, num_classes=num_classes)
X = X.reshape(X.shape[0], X.shape[1], 1)  

scaler = StandardScaler()
for i in range(X.shape[2]):
    X[:,:,i] = scaler.fit_transform(X[:,:,i])

model = Sequential()  
model.add(LSTM(units=128, activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.5)) 
model.add(LSTM(units=64, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_encoded, epochs=10, batch_size=64) 

joblib.dump(scaler, 'rnn_scaler.pkl')
model.save('rnn.h5')