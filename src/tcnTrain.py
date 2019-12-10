from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, Flatten,LocallyConnected1D
from keras.models import  Model, Sequential
from numpy.random import seed
from tcn import TCN
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

commands = open("timeSeries.py").read()
exec(commands)

model_path    = '../models/model-cnn.h5'

X = []
Y = []
for i in range(1,6,1):
    indices = data['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled[indices][:,time_series_feature_columns], np.array(data.loc[indices,'converged'])):
        X.append(x)
    for y in InputToTimeSeries(data_scaled[indices][:,target_columns], np.array(data.loc[indices,'converged'])):
        Y.append(y[-1])
X = np.array(X)
Y = np.array(Y)

X_val = []
Y_val = []
for i in range(7,8,1):
    indices = data_noiter['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled_noiter[indices][:, time_series_feature_columns]):
        X_val.append(x)
    for y in  InputToTimeSeries(data_scaled_noiter[indices][:, target_columns]):
        Y_val.append(y[-1])
X_val = np.array(X_val)
Y_val = np.array(Y_val)

i = Input(shape=(time_series_steps, len(time_series_feature_columns)))
o = TCN(nb_filters=32, kernel_size=1, activation='sigmoid', name='tcn_1')(i)
o = Dropout(0.1) (o)
#o = TCN(nb_filters=64, kernel_size=1, activation='sigmoid', name='tcn_2')(o)
#o = Dropout(0.1)(o)
o = Flatten()(o)
o = Dense(10, activation='sigmoid') (o)
o = Dropout(0.1) (o)
o = Dense(2) (o)

model = Model(inputs = [i], outputs=[o])
model.compile(loss='mse', optimizer='adam')
with open('../results/summary_cnn.txt','w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

history = model.fit(X, Y, epochs = 500, batch_size = 32, validation_data=(X_val, Y_val),
                    callbacks=[ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)])

pd.DataFrame(history.history).to_csv("../results/train-cnn.csv")
