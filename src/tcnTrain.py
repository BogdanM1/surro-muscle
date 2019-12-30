from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Dropout, GRU, MaxPooling1D, Flatten
from keras.models import  Model, Sequential
from tcn import TCN
from numpy.random import seed
from tensorflow import set_random_seed 

seed(1)
set_random_seed(2)

commands = open("timeSeries.py").read()
exec(commands)

model_path    = '../models/model-tcn.h5'

X = []
Y = []
for i in range(1,14,1):
    indices = data['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled[indices][:,time_series_feature_columns], np.array(data.loc[indices,'converged'])):
        X.append(x)
    for y in InputToTimeSeries(data_scaled[indices][:,target_columns], np.array(data.loc[indices,'converged'])):
        Y.append(y[-1])
X = np.array(X)
Y = np.array(Y)

X_val = []
Y_val = []
for i in range(14,15,1):
    indices = data_noiter['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled_noiter[indices][:, time_series_feature_columns]):
        X_val.append(x)
    for y in  InputToTimeSeries(data_scaled_noiter[indices][:, target_columns]):
        Y_val.append(y[-1])
X_val = np.array(X_val)
Y_val = np.array(Y_val)

i = Input(shape=(time_series_steps, len(time_series_feature_columns)))
o = TCN(nb_filters=8, kernel_size=1, activation='wavenet', name='tcn_1')(i)
#o = Dropout(0.1)(o)
o = TCN(nb_filters=16, activation='wavenet', name='tcn_2')(o)
#o = Dropout(0.1)(o)
o = Flatten()(o)
o = Dense(128, activation='sigmoid')(o)
o = Dropout(0.1)(o)
o = Dense(2) (o)
model = Model(inputs = [i], outputs=[o])
model.compile(loss=huber_loss(), optimizer='adam')

history = model.fit(X, Y, epochs = 1000, batch_size = 512, validation_data=(X_val, Y_val),
                    callbacks=[ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)])
pd.DataFrame(history.history).to_csv("../results/train-cnn.csv")

