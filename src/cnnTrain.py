from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.models import  Sequential
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

commands = open("timeSeries.py").read()
exec(commands)

model_path    = '../models/model-cnn.h5'

X = []
Y = []
for i in range(1,15,2):
    indices = data['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled[indices][:,time_series_feature_columns], time_series_steps, np.array(data.loc[indices,'converged'])):
        X.append(x)
    for y in InputToTimeSeries(data_scaled[indices][:,target_columns], time_series_steps, np.array(data.loc[indices,'converged'])):
        Y.append(y[-1])
X = np.array(X)
Y = np.array(Y)

X_val = []
Y_val = []
for i in range(2,15,2):
    indices = data_noiter['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled_noiter[indices][:, time_series_feature_columns], time_series_steps):
        X_val.append(x)
    for y in  InputToTimeSeries(data_scaled_noiter[indices][:, target_columns], time_series_steps):
        Y_val.append(y[-1])
X_val = np.array(X_val)
Y_val = np.array(Y_val)

model = Sequential()
model.add(Conv1D(filters=60, kernel_size=1, input_shape = (time_series_steps, len(time_series_feature_columns)), activation='sigmoid'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(30, activation='sigmoid'))
model.add(Dropout(0.15))
model.add(Dense(2))
model.compile(loss='mse', optimizer='adam')
with open('../results/summary_cnn.txt','w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

history = model.fit(X, Y, epochs = 500, batch_size = 32, validation_data=(X_val, Y_val),
                    callbacks=[ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)])

pd.DataFrame(history.history).to_csv("../results/train-cnn.csv")

