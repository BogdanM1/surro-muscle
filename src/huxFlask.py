from flask import Flask
from flask import request
import tensorflow as tf
from keras.models import load_model
import numpy as np
from sklearn.externals import joblib
import os

app = Flask(__name__)

lregr_path = '../models/regr.sav'
lregr_model = joblib.load(lregr_path)

models_directory = "../models/"
models = {}
model_path  = os.path.join(models_directory,"_model-new.h5")
session = tf.Session()
graph = tf.get_default_graph()
with graph.as_default(), session.as_default():
    model = load_model(model_path)
models['_model']  = {}
models['_model']['model'] = model
models['_model']['session'] = session
models['_model']['graph'] = graph

commands = open("load_data.py").read()
exec(commands)

@app.route('/save_net', methods = ['POST'])
def loadNet():
  global models
  params = request.form.to_dict()
  netname = params['netname']
  model_export = request.files['network']
  model_path = os.path.join(models_directory, netname + ".h5")
  model_export.save(model_path)
  graph = tf.get_default_graph()
  session = tf.Session()
  with graph.as_default(), session.as_default():
      model = load_model(model_path)
  models[netname] = {}
  models[netname]['model'] = model
  models[netname]['session'] = session
  models[netname]['graph'] = graph
  return "OK"

@app.route('/sigdsig-regr', methods=['POST'])
def regr_prediction():
  params     = request.form.to_dict()

  activation = params['activation'].split(',')
  stretch_prev = params['stretch_prev'].split(',')
  stretch = params['stretch'].split(',')
  sigma_prev = params['sigma_prev'].split(',')
  delta_sigma_prev = params['delta_sigma_prev'].split(',')

  activation = np.array(activation, dtype='f')
  stretch_prev = np.array(stretch_prev, dtype='f')
  stretch = np.array(stretch, dtype='f')
  sigma_prev = np.array(sigma_prev, dtype='f')
  delta_sigma_prev = np.array(delta_sigma_prev, dtype='f')
  input_matrix = np.column_stack((activation, stretch_prev, stretch, sigma_prev, delta_sigma_prev))
  predicted = lregr_model.predict(input_matrix)
  sigma_predicted = predicted[:, 0]
  dsigma_predicted = predicted[:, 1]
  result = ""
  count = len(sigma_predicted)
  for i in range(count):
    result += str(sigma_predicted[i]) + "#" + str(dsigma_predicted[i]) + ","
  return result

@app.route('/sigdsig', methods=['POST'])
def prediction():
  global models

  params     = request.form.to_dict()
  netname    = params['netname']
  model      = models[netname]['model']
  graph      = models[netname]['graph']
  session    = models[netname]['session']

  activation = params['activation'].split(',')
  stretch_prev = params['stretch_prev'].split(',')
  stretch = params['stretch'].split(',')
  sigma_prev = params['sigma_prev'].split(',')
  delta_sigma_prev = params['delta_sigma_prev'].split(',')

  activation = np.array(activation, dtype='f')
  stretch_prev = np.array(stretch_prev, dtype='f')
  stretch = np.array(stretch, dtype='f')
  sigma_prev = np.array(sigma_prev, dtype='f')
  delta_sigma_prev = np.array(delta_sigma_prev, dtype='f')
  input_matrix = np.column_stack((activation, stretch_prev, stretch, sigma_prev, delta_sigma_prev))
  input_matrix_scaled = (input_matrix - scaler.data_min_[np.r_[feature_columns]]) / scaler.data_range_[np.r_[feature_columns]]

  with graph.as_default(), session.as_default():
      predicted = model.predict(input_matrix_scaled)
  sigma_predicted = predicted[:, 0] * scaler.data_range_[target_columns[0]] + scaler.data_min_[target_columns[0]]
  dsigma_predicted = predicted[:, 1] * scaler.data_range_[target_columns[1]] + scaler.data_min_[target_columns[1]]

  result = ""
  count = len(sigma_predicted)
  for i in range(count):
    result += str(sigma_predicted[i]) + "#" + str(dsigma_predicted[i]) + ","
  return result

lstm_steps = 3
lstm_input = np.zeros((1, lstm_steps, 2))
lstm_feature_columns = np.array([1, 3])
start = True
@app.route('/start-lstm', methods=['POST'])
def start():
  global start
  start = True
  return "OK"

@app.route('/sigdsig_lstm', methods=['POST'])
def prediction_lstm():
  global start
  global models

  params     = request.form.to_dict()
  netname    = params['netname']
  model      = models[netname]['model']
  graph      = models[netname]['graph']
  session    = models[netname]['session']
  
  activation = params['activation'].split(',')
  stretch = params['stretch'].split(',')
  activation = np.array(activation, dtype='f')
  stretch = np.array(stretch, dtype='f')
  input_matrix = np.column_stack((activation, stretch))
  input_matrix_scaled = (input_matrix - scaler.data_min_[np.r_[lstm_feature_columns]]) / scaler.data_range_[np.r_[lstm_feature_columns]]

  nqp = len(activation)
  predicted = np.zeros((nqp, 2))
  with graph.as_default(), session.as_default():
      for i in range(0, nqp):
        one_row = input_matrix_scaled[i, :]
        if start:
          start = False
          for j in range(0, lstm_steps):
            lstm_input[0, j, :] = one_row
        else:
          for j in range(0, lstm_steps - 1):
            lstm_input[0, j, :] = lstm_input[0, j + 1, :]
          lstm_input[0, lstm_steps - 1,] = one_row
        lstm_predicted = model.predict(lstm_input)
        predicted[i, :] = lstm_predicted[0, lstm_steps-1, :]

  sigma_predicted  = predicted[:, 0] * scaler.data_range_[target_columns[0]] + scaler.data_min_[target_columns[0]]
  dsigma_predicted = predicted[:, 1] * scaler.data_range_[target_columns[1]] + scaler.data_min_[target_columns[1]]

  result = ""
  count = len(sigma_predicted)
  for i in range(count):
    result += str(sigma_predicted[i]) + "#" + str(dsigma_predicted[i]) + ","
  return result



if __name__ == '__main__':
   app.run(port = 8000, host = "147.91.204.14", debug = True)
