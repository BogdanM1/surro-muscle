import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation, LeakyReLU
from keras import optimizers
from keras_radam import RAdam
from keras import backend as K
import itertools

K.set_floatx('float32')

feature_columns = [1, 2, 3, 4, 5, 6]
target_columns  = [7, 8]

data = pd.read_csv("../data/dataMexie.csv")
data_noiter = pd.read_csv("../data/dataMexieNoIter.csv")
'''
data_large  = pd.read_csv("../data/dataMexieLargeModel.csv")
data_noiter_large = pd.read_csv("../data/dataMexieLargeModelNoIter.csv")

data = data.append(data_large)
data_noiter = data_noiter.append(data_noiter_large) 
del data_large
del data_noiter_large
'''

scaler = MinMaxScaler(feature_range=(0,1))

chunk_size=10000
for i in itertools.chain(np.setdiff1d(range(1,45),range(4,45,4))): 
    indices = data['testid'].isin([i])
    scaler.partial_fit(data[indices])

for start in range(0, data.shape[0], chunk_size):
  df_subset = data.iloc[start:start + chunk_size]
  if(start==0):
      data_scaled = scaler.transform(df_subset)
  else:
      data_scaled = np.append(data_scaled, scaler.transform(df_subset), axis=0)

for start in range(0, data_noiter.shape[0], chunk_size):
  df_subset = data_noiter.iloc[start:start + chunk_size]  
  if(start==0):
      data_scaled_noiter = scaler.transform(df_subset)
  else:
      data_scaled_noiter = np.append(data_scaled_noiter, scaler.transform(df_subset), axis=0)  

# huber 
def huber_loss(tolerance=.01):
    def huber(y,y_pred):
        error = y - y_pred
        is_small_error = tf.abs(error) < tolerance
        squared_loss = tf.square(error) / 2 
        linear_loss = tolerance*tf.abs(error) - tolerance*tolerance*0.5 
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber

# smape
def smape(true,predicted):
    epsilon = 0.1
    summ = K.maximum(K.abs(true) + K.abs(predicted) + epsilon, 0.5 + epsilon)
    err = K.abs(predicted - true) / summ * 2.0
    return err    

