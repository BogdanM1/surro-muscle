from flask import Flask
from flask import request
import tensorflow as tf
from keras.models import load_model
import numpy as np
from sklearn.externals import joblib
import os
import sys

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

commands = open("LSTMfeatures.py").read()
exec(commands)

lstm_start = True
nqp = 4
lstm_input  = np.zeros((1, lstm_steps, len(lstm_feature_columns)))
model_path  = os.path.join(models_directory,"_model-lstm.h5")
with graph.as_default(), session.as_default():
    model = load_model(model_path)
models['lstm_model']  = {}
models['lstm_model']['model'] = model
models['lstm_model']['session'] = session
models['lstm_model']['graph'] = graph

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

  activation_prev = params['activation_prev'].split(',')
  activation = params['activation'].split(',')
  stretch_prev = params['stretch_prev'].split(',')
  stretch = params['stretch'].split(',')
  sigma_prev = params['sigma_prev'].split(',')
  delta_sigma_prev = params['delta_sigma_prev'].split(',')

  activation_prev = np.array(activation_prev, dtype='f')
  activation = np.array(activation, dtype='f')
  stretch_prev = np.array(stretch_prev, dtype='f')
  stretch = np.array(stretch, dtype='f')
  sigma_prev = np.array(sigma_prev, dtype='f')
  delta_sigma_prev = np.array(delta_sigma_prev, dtype='f')
  input_matrix = np.column_stack((activation_prev, activation, stretch_prev, stretch, sigma_prev, delta_sigma_prev))
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

  activation_prev = params['activation_prev'].split(',')
  activation = params['activation'].split(',')
  stretch_prev = params['stretch_prev'].split(',')
  stretch = params['stretch'].split(',')
  sigma_prev = params['sigma_prev'].split(',')
  delta_sigma_prev = params['delta_sigma_prev'].split(',')

  activation_prev = np.array(activation_prev, dtype='f')
  activation = np.array(activation, dtype='f')
  stretch_prev = np.array(stretch_prev, dtype='f')
  stretch = np.array(stretch, dtype='f')
  sigma_prev = np.array(sigma_prev, dtype='f')
  delta_sigma_prev = np.array(delta_sigma_prev, dtype='f')
  input_matrix = np.column_stack((activation_prev,activation, stretch_prev, stretch, sigma_prev, delta_sigma_prev))
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

@app.route('/start-lstm', methods=['GET'])
def start():
  global lstm_start, nqp, lstm_input
  lstm_start = True
  lstm_input  = np.zeros((nqp, lstm_steps, len(lstm_feature_columns)))
  return "OK"

@app.route('/sigdsig-lstm', methods=['POST'])
def prediction_lstm():
  global lstm_start
  global models
  global nqp

  params     = request.form.to_dict()
  netname    = params['netname']
  model      = models[netname]['model']
  graph      = models[netname]['graph']
  session    = models[netname]['session']
  
  activation = params['activation'].split(',')
  stretch    = params['stretch'].split(',')
  sigma_prev = params['sigma_prev'].split(',')
  delta_sigma_prev = params['delta_sigma_prev'].split(',')
  converged = params['converged']
  
  activation = np.array(activation, dtype='f')
  stretch = np.array(stretch, dtype='f')
  sigma_prev = np.array(sigma_prev, dtype='f')
  delta_sigma_prev = np.array(delta_sigma_prev, dtype='f')  
  input_matrix = np.column_stack((activation, stretch, sigma_prev, delta_sigma_prev))
  input_matrix_scaled = (input_matrix - scaler.data_min_[np.r_[lstm_feature_columns]]) / scaler.data_range_[np.r_[lstm_feature_columns]]

  predicted = np.zeros((nqp, 2))
  if lstm_start:
      lstm_start = False
      for i in range(0, nqp):
        for j in range(0, lstm_steps):
          lstm_input[i, j, :] = input_matrix_scaled[i, :]
  else:
      if(int(converged) == 1):
        for i in range(0, nqp):
          for j in range(0, lstm_steps - 1):
              lstm_input[i, j, :] = lstm_input[i, j + 1, :]
      for i in range(0, nqp):
        lstm_input[i, lstm_steps - 1,] = input_matrix_scaled[i, :]

  with graph.as_default(), session.as_default():
    lstm_predicted = model.predict(lstm_input)
    predicted[:, :] = lstm_predicted[:, lstm_steps-1, :]

  sigma_predicted  = predicted[:, 0] * scaler.data_range_[target_columns[0]] + scaler.data_min_[target_columns[0]]
  dsigma_predicted = predicted[:, 1] * scaler.data_range_[target_columns[1]] + scaler.data_min_[target_columns[1]]

  result = ""
  count = len(sigma_predicted)
  for i in range(count):
    result += str(sigma_predicted[i]) + "#" + str(dsigma_predicted[i]) + ","
  return result


app.run(port = 8000, host = "147.91.204.14", debug = True)
