import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation, LeakyReLU
from keras import optimizers
from keras_radam import RAdam
from keras import backend as K

K.set_floatx('float32')

feature_columns = [1, 2, 3, 4, 5, 6]
target_columns  = [7, 8]
data          = pd.read_csv("../data/dataMexie.csv")
data_noiter   = pd.read_csv("../data/dataMexieNoIter.csv")

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(data)

# stress scale equally input and output 
scaler.data_min_[5] = min(scaler.data_min_[5], scaler.data_min_[7])
scaler.data_min_[7] = scaler.data_min_[5]
#
scaler.data_max_[5] = max(scaler.data_max_[5], scaler.data_max_[7])
scaler.data_max_[7] = scaler.data_max_[5]
#
scaler.data_range_[5] = scaler.data_max_[5] - scaler.data_min_[5]
scaler.data_range_[7] = scaler.data_range_[7]

# stress derivative scale equally input and output 
scaler.data_min_[6] = min(scaler.data_min_[6], scaler.data_min_[8])
scaler.data_min_[8] = scaler.data_min_[8]
#
scaler.data_max_[6] = max(scaler.data_max_[6], scaler.data_max_[8])
scaler.data_max_[8] = scaler.data_max_[6]
#
scaler.data_range_[6] = scaler.data_max_[6] - scaler.data_min_[6]
scaler.data_range_[8] = scaler.data_range_[6]
#########################################
data_scaled        = scaler.transform(data)
data_scaled_noiter = scaler.transform(data_noiter)

# huber 
def huber_loss(tolerance=.01):
    def huber(y,y_pred):
        error = y - y_pred
        is_small_error = tf.abs(error) < tolerance
        squared_loss = tf.square(error) / 2 
        linear_loss = tolerance*tf.abs(error) - tolerance*tolerance*0.5 
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber

