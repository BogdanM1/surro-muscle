from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Dropout, Flatten, GRU
from keras.models import  Model, Sequential
from mytcn import TCN
from numpy.random import seed
from tensorflow import set_random_seed 
from keras_radam import RAdam
import keras.initializers

_seed = 137
seed(_seed)
set_random_seed(_seed)

commands = open("timeSeries.py").read()
exec(commands)
model_path    = '../models/model-gru-tcn.h5'

X = []
Y = []
for i in itertools.chain(np.setdiff1d(range(1,45),range(4,45,4))): 
    indices = data['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled[indices][:,time_series_feature_columns], np.array(data.loc[indices,'converged'])):
        X.append(x)
    for y in InputToTimeSeries(data_scaled[indices][:,target_columns], np.array(data.loc[indices,'converged'])):
        Y.append(y[-1])
    data = data[indices != True]
    data_scaled = data_scaled[indices != True]
X = np.array(X)
Y = np.array(Y)

X_val = []
Y_val = []
for i in itertools.chain(range(4,45,4)): 
    indices = data_noiter['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled_noiter[indices][:, time_series_feature_columns]):
        X_val.append(x)
    for y in  InputToTimeSeries(data_scaled_noiter[indices][:, target_columns]):
        Y_val.append(y[-1])
    data_noiter = data_noiter[indices != True]
    data_scaled_noiter = data_scaled_noiter[indices != True]        
X_val = np.array(X_val)
Y_val = np.array(Y_val)

lecun_normal = keras.initializers.lecun_normal(seed=_seed)
orthogonal = keras.initializers.Orthogonal(seed=_seed)
glorot_uniform = keras.initializers.glorot_uniform(seed=_seed)

i = Input(shape=(time_series_steps, len(time_series_feature_columns)), name='input_layer')
o = GRU(128, return_sequences=True, kernel_initializer=orthogonal, recurrent_initializer=orthogonal, recurrent_dropout=.0, name='gru_layer1')(i) 
#o = Dropout(.2)(o)
o = TCN(nb_filters=64, kernel_size=4, dilations=[1,2,4], activation='selu', kernel_initializer=lecun_normal, use_skip_connections=False, name='tcn_layer1')(o) 
o = Flatten()(o)
o = Dense(2, name='output_layer') (o)

model = Model(inputs = [i], outputs=[o])
model.compile(loss=huber_loss(), optimizer=RAdam(learning_rate=1e-4, amsgrad=True))
print(model.summary())
history = model.fit(X, Y, epochs = 30000, batch_size = 16384, validation_data = (X_val, Y_val), verbose=2,
	            callbacks = [ModelCheckpoint(model_path, monitor = 'val_loss', save_best_only = True),
                    EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=300)])
pd.DataFrame(history.history).to_csv("../results/train-grutcn.csv")

