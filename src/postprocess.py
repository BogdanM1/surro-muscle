import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.externals import joblib
import os

commands = open("load_data.py").read()
exec(commands)

commands = open("LSTMfeatures.py").read()
exec(commands)

num_tests = 15

showTestData  = False
writeDataResults = True
writeDynamicResults = True

use_nnet = True
use_lstm  = True
use_lregr = False

model_path      = '../models/_model-new.h5'
lstm_model_path      = '../models/_model-lstm.h5'
regr_model_path = '../models/regr.sav'

if(use_lregr):
	model = joblib.load(regr_model_path)
if(use_nnet):
    model = load_model(lstm_model_path) if(use_lstm) else load_model(model_path)

results_dir = '../results/'
for file_name in os.listdir(results_dir):
    if file_name.endswith('.png') or (file_name.startswith('data') and file_name.endswith('.csv')):
        os.unlink(results_dir + file_name)

def drawGraph(x, y, name, unit, testid):
    global  results_dir
    plt.figure(figsize=(5, 4), dpi=80)
    plt.plot(x, y)
    plt.xlabel('Time [s]')
    plt.ylabel(name + ' ' + unit)
    plt.title('Test ' + str(testid) + ' - ' + name)
    plt.tight_layout()
    plt.savefig(results_dir + name + str(testid) + '.png')
    plt.close()

def drawGraphRes(x, y1, y2, name1, name2, title, testid):
    global results_dir
    plt.figure(figsize=(5, 4), dpi=80)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.xlabel('Time [s]')
    plt.ylabel(title + ' [pN/nm^2]')
    plt.title('Test ' + str(testid) + ' - ' + title, loc = 'left')
    plt.legend([name1, name2], loc='lower right',  mode="expand", bbox_to_anchor=(0.35, 1.2), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(results_dir + title + str(testid) + '.png')
    plt.close()

def drawTestData(data):
    for i in range(1, num_tests + 1):
        data_test = data.query('testid==' + str(i))

        time = np.array(data_test['time'])
        activation = np.array(data_test['activation'])
        stretch = np.array(data_test['stretch'])
        sigma = np.array(data_test['sigma'])
        delta_sigma = np.array(data_test['delta_sigma'])

        drawGraph(time, activation, 'Activation', '[%]',i)
        drawGraph(time, stretch, 'Stretch', '',i)
        drawGraph(time, sigma, 'Sigma', '[pN/nm^2]',i)
        drawGraph(time, delta_sigma, 'Delta sigma', '[pN/nm^2]',i)

def list_to_num(numList):         
    s = map(str, numList)   
    s = ''.join(s)          
    s = int(s)              
    return s
	
def drawTestResults():
    global results_dir
    for file_name in os.listdir(results_dir):
        if not file_name.startswith('data') and not file_name.startswith('dynamic'):
            continue
        data = pd.read_csv(results_dir + file_name)
        time = np.array(data['time'])
        sigma = np.array(data['sigma'])
        delta_sigma = np.array(data['delta_sigma'])
        sigma_pred = np.array(data['sigma pred'])
        delta_sigma_pred = np.array(data['delta_sigma pred'])
        testid = list_to_num([int(s) for s in file_name if s.isdigit()])
        if file_name.startswith('data'):
            drawGraphRes(time, sigma, sigma_pred, 'original', 'predicted', 'Stress', testid)
            drawGraphRes(time, delta_sigma, delta_sigma_pred, 'original', 'predicted','Stress derivative', testid)
        else:
            drawGraphRes(time, sigma, sigma_pred, 'original', 'predicted', 'Stress (dynamic)', testid)
            drawGraphRes(time, delta_sigma, delta_sigma_pred, 'original', 'predicted', 'Stress derivative (dynamic)', testid)
if(writeDataResults):
	for i in range(0,num_tests):
		indices       = data_noiter.index[data_noiter['testid'] == (i+1)].tolist()
		original_data = np.array(data_noiter)[indices, :]
		pred_data     = np.array(data_scaled_noiter)[indices, :] if(use_nnet) else np.array(data_noiter)[indices, :]
		if(use_lstm):
		    pred_data = pred_data[:, lstm_feature_columns]
		    lstm_prediction = model.predict(reshapeInputLSTM(pred_data, lstm_steps))
		    prediction =  np.zeros((len(lstm_prediction), len(target_columns)))
		else:
		    pred_data = pred_data[:, feature_columns]
		    prediction    = model.predict(pred_data)
		if(use_lstm):
		    for itarg in range(0, len(target_columns)):
		        prediction[:, itarg] = lstm_prediction[:, lstm_steps - 1, itarg]
		if(use_nnet):
		    for itarg in range(0, len(target_columns)):
		        prediction[:, itarg] = prediction[:, itarg] * scaler.data_range_[target_columns[itarg]]  + scaler.data_min_[target_columns[itarg]]
		nlen = len(prediction)
		df = pd.DataFrame(data = { 'time': original_data[0:nlen,0],
                                   'sigma': original_data[0:nlen,target_columns[0]],
                                   'delta_sigma': original_data[0:nlen,target_columns[1]],
                                   'sigma pred': prediction[:, 0],
                                   'delta_sigma pred': prediction[:,1]})
		df.to_csv(results_dir + 'data_pred_test' + str(i+1) + '.csv', index=False)


if(writeDynamicResults):
	for i in range(1,num_tests):
	    try:
	        indices       = data_noiter.index[data_noiter['testid'] == (i+1)].tolist()
	        original_data = np.array(data_noiter)[indices, :]
	        pred_data = pd.read_csv(results_dir + "surroHuxley"+str(i+1)+".csv", sep='\s*,\s*', engine='python')
	        pred_data = pred_data.iloc[::4, :]
	        sigma_pred  = pred_data['sigma']
	        dsigma_pred = pred_data['delta_sigma']
	        df = pd.DataFrame(data = { 'time': original_data[:,0],
                                       'sigma': original_data[:,target_columns[0]],
                                       'delta_sigma': original_data[:,target_columns[1]],
                                       'sigma pred': sigma_pred,
                                       'delta_sigma pred': dsigma_pred})
	        df.to_csv(results_dir + 'dynamic_pred_test' + str(i+1) + '.csv', index=False)
	    except:
	        print("Error during processing test No. " + str(i+1))

if(showTestData): drawTestData(data_noiter)
drawTestResults()
