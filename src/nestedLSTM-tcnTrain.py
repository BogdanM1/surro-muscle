from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, Dropout, Flatten, Bidirectional
from keras.models import  Model, Sequential
from mytcn import TCN
from numpy.random import seed
from nested_lstm import NestedLSTM


commands = open("initialize.py").read()
exec(commands)
model_path    = '../models/model.h5'

_seed = 137
seed(_seed)
tf.random.set_seed(_seed)

from adaptive import AdaptiveLossFunction
adaptive = AdaptiveLossFunction(2,tf.float32,scale_lo=1e-10,scale_init=1e-4)   
 
def adaptive_jonbarron_loss(y_true, y_pred): 
   return adaptive.__call__(K.abs(y_pred-y_true))
   
def adaptive_jonbarron_loss_td(y_true, y_pred): 
    y_true_curr = y_true[:,:2]
    y_true_prev = y_true[:,-2:]
    y_true_diff = K.abs(y_true_curr - y_true_prev) + 1e-10
    return y_true_diff*1e+4*adaptive.__call__(K.abs(y_pred - y_true_curr))

loss = adaptive_jonbarron_loss_td


X = []
Y = []
for i in itertools.chain(np.setdiff1d(range(ntrains,ntraine),range(ntrains+3,ntraine,4))):  
    indices = data['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled[indices][:,time_series_feature_columns], np.array(data.loc[indices,'converged'])):
        X.append(x)
    for y in InputToTimeSeries(data_scaled[indices][:,target_columns], np.array(data.loc[indices,'converged'])):
        Y.append(y[-1])
    data = data[indices != True]
    data_scaled = data_scaled[indices != True]   


X_val = []
Y_val = []
for i in itertools.chain(range(ntrains+3,ntraine,4)): 
    indices = data['testid'].isin([i])
    for x in InputToTimeSeries(data_scaled[indices][:, time_series_feature_columns], np.array(data.loc[indices,'converged'])):
        X_val.append(x)
    for y in  InputToTimeSeries(data_scaled[indices][:, target_columns], np.array(data.loc[indices,'converged'])):
        Y_val.append(y[-1])
    data = data[indices != True]
    data_scaled = data_scaled[indices != True]   


values_prev = [sample[-1][-2:] for sample in X]
values_prev_val = [sample[-1][-2:] for sample in X_val]

X = np.array(X)
Y = np.array(Y)         
Y = np.append(Y, values_prev, axis = 1)


X_val = np.array(X_val)
Y_val = np.array(Y_val)
Y_val = np.append(Y_val, values_prev_val, axis = 1)

lecun_normal = keras.initializers.lecun_normal(seed=_seed)
orthogonal = keras.initializers.Orthogonal(seed=_seed)
glorot_uniform = keras.initializers.glorot_uniform(seed=_seed)

i = Input(shape=(time_series_steps, len(time_series_feature_columns)), name='input_layer')
o = NestedLSTM(128, depth=6, return_sequences=True, activation='selu', kernel_initializer=lecun_normal, recurrent_initializer=orthogonal, name='lstm1')(i) 
o = TCN(nb_filters=128, kernel_size=4, dilations=[1,2,4], activation='selu', kernel_initializer=lecun_normal, use_skip_connections=False, name='tcn1')(o) 
o = Flatten()(o)
o = Dense(2, name='output_layer')(o)

model = Model(inputs = [i], outputs=[o])
model.compile(loss=loss, optimizer=optimizer)
print(model.summary())


history = model.fit(X, Y, epochs = 50000, batch_size = 16384, validation_data = (X_val, Y_val), verbose=2, 
	            callbacks = [ModelCheckpoint(model_path, monitor = 'val_loss', save_best_only = True),
                    EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=300)])
pd.DataFrame(history.history).to_csv("../results/train-grutcn.csv")


