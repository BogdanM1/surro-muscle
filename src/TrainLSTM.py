from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.models import  Sequential
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

commands = open("load_data.py").read()
exec(commands)

commands = open("LSTMfeatures.py").read()
exec(commands)

def reshapeInputLSTM (data, lstm_steps):
  data_count = len(data) - lstm_steps + 1
  nfeatures = len(data[0])
  outdata   = np.empty([data_count, lstm_steps, nfeatures])
  for i in range(0,data_count):
    chunk         = data[i:(i+lstm_steps), :]
    outdata[i,:,:]  = chunk
  return (outdata)


model_path    = '../models/_model-lstm.h5'
training_data = data_scaled[data['testid'].isin([1, 2, 3, 4, 5, 6])]
X = training_data[:, lstm_feature_columns]
Y = training_data[:, target_columns]

X = reshapeInputLSTM(X, lstm_steps)
Y = reshapeInputLSTM(Y, lstm_steps)

val_data = data_scaled[data['testid'].isin([7])]
X_val = val_data[:, lstm_feature_columns]
Y_val = val_data[:, lstm_feature_columns]

X_val = reshapeInputLSTM(X_val, lstm_steps)
Y_val = reshapeInputLSTM(Y_val, lstm_steps)

model = Sequential()
model.add(LSTM(60, input_shape = (lstm_steps, len(lstm_feature_columns)), activation='sigmoid', return_sequences=True))
model.add(Dropout(0.1))
model.add(Dense(30, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(2))
model.compile(loss='mse', optimizer='adam')

history = model.fit(X, Y, epochs = 500, batch_size = 32, validation_data=(X_val, Y_val),
                    callbacks=[ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)])

plt.plot(history.history['loss'])
plt.savefig('../results/train-lstm.png')
plt.close()

