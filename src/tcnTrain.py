from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Dropout, Flatten, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.models import  Model, Sequential
from mytcn import TCN
from numpy.random import seed
from keras_self_attention import SeqSelfAttention

commands = open("initialize.py").read()
exec(commands)
model_path    = '../models/model.h5'

_seed = 137
seed(_seed)
tf.random.set_seed(_seed)

X = []
Y = []
for i in range(20,29,1):
    indices = data['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled[indices][:,time_series_feature_columns], np.array(data.loc[indices,'converged'])):
        X.append(x)
    for y in InputToTimeSeries(data_scaled[indices][:,target_columns], np.array(data.loc[indices,'converged'])):
        Y.append(y[-1])
X = np.array(X)
Y = np.array(Y)

X_val = []
Y_val = []
for i in range(16,20,1):
    indices = data_noiter['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled_noiter[indices][:, time_series_feature_columns]):
        X_val.append(x)
    for y in  InputToTimeSeries(data_scaled_noiter[indices][:, target_columns]):
        Y_val.append(y[-1])
X_val = np.array(X_val)
Y_val = np.array(Y_val)

i = Input(shape=(time_series_steps, len(time_series_feature_columns)))
o = TCN(nb_filters = 8, kernel_size = 4, dilations  = [1,2,4], activation = 'wavenet',
        return_sequences= True, use_skip_connections = False, use_layer_norm = False)(i) 
o = Flatten()(o)
o = Dense(2) (o)

model = Model(inputs = [i], outputs=[o])
model.compile(loss=loss, optimizer=optimizer)
history = model.fit(X, Y, epochs = 3000, batch_size = 1024, validation_data=(X_val, Y_val),
                    callbacks=[ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)])
pd.DataFrame(history.history).to_csv("../results/train-tcn.csv")

