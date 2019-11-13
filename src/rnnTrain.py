from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, SimpleRNN
from keras.models import  Sequential
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

commands = open("timeSeries.py").read()
exec(commands)

model_path    = '../models/model-rnn.h5'

X = []
Y = []
for i in range(1,12,1):
    indices = data['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled[indices][:,time_series_feature_columns], np.array(data.loc[indices,'converged'])):
        X.append(x)
    for y in InputToTimeSeries(data_scaled[indices][:,target_columns], np.array(data.loc[indices,'converged'])):
        Y.append(y)
X = np.array(X)
Y = np.array(Y)

X_val = []
Y_val = []
for i in range(12,15,1):
    indices = data['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled[indices][:, time_series_feature_columns], np.array(data.loc[indices,'converged'])):
        X_val.append(x)
    for y in  InputToTimeSeries(data_scaled[indices][:, target_columns], np.array(data.loc[indices,'converged'])):
        Y_val.append(y)
X_val = np.array(X_val)
Y_val = np.array(Y_val)

model = Sequential()
model.add(SimpleRNN(60, input_shape = (time_series_steps, len(time_series_feature_columns)), activation='sigmoid', return_sequences=True))
model.add(Dropout(0.1))
model.add(SimpleRNN(40, activation='sigmoid', return_sequences=True))
model.add(Dropout(0.1))
model.add(SimpleRNN(20, activation='sigmoid', return_sequences=True))
model.add(Dropout(0.1))
model.add(Dense(2))
model.compile(loss='mse', optimizer='adam')
with open('../results/summary_rnn.txt','w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

history = model.fit(X, Y, epochs = 500, batch_size = 32, validation_data=(X_val, Y_val),
                    callbacks=[ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)])

pd.DataFrame(history.history).to_csv("../results/train-rnn.csv")
