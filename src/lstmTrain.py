from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.models import  Sequential
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

commands = open("LSTMfeatures.py").read()
exec(commands)

model_path    = '../models/_model-lstm.h5'

X = []
Y = []
for i in range(1,7):
    indices = data['testid'].isin([i])
    for x in reshapeInputLSTM(data_scaled[indices][:,lstm_feature_columns], lstm_steps, np.array(data.loc[indices,'converged'])):
        X.append(x)
    for y in reshapeInputLSTM(data_scaled[indices][:,target_columns], lstm_steps, np.array(data.loc[indices,'converged'])):
        Y.append(y)
X = np.array(X)
Y = np.array(Y)

X_val = []
Y_val = []
for i in range(7, 8):
    indices = data_noiter['testid'].isin([i])
    for x in reshapeInputLSTM(data_scaled_noiter[indices][:, lstm_feature_columns], lstm_steps):
        X_val.append(x)
    for y in  reshapeInputLSTM(data_scaled_noiter[indices][:, target_columns], lstm_steps):
        Y_val.append(y)
X_val = np.array(X_val)
Y_val = np.array(Y_val)

model = Sequential()
model.add(LSTM(60, input_shape = (lstm_steps, len(lstm_feature_columns)), activation='sigmoid', return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(30, activation='sigmoid', return_sequences=True))
model.add(Dropout(0.1))
model.add(Dense(2))
model.compile(loss='mse', optimizer='adam')
with open('../results/summary_lstm.txt','w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

history = model.fit(X, Y, epochs = 500, batch_size = 32, validation_data=(X_val, Y_val),
                    callbacks=[ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)])

pd.DataFrame(history.history).to_csv("../results/train-lstm.csv")

