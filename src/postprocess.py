import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
import math
import os
from keras_multi_head import MultiHeadAttention
from keras_self_attention import SeqSelfAttention
from keras_radam import RAdam
from keras_adabound import AdaBound
from keras_lookahead import Lookahead

commands = open("timeSeries.py").read()
exec(commands)

num_tests = 15

showTestData  = False
writeDataResults = True
writeDynamicResults = True

model_path      = '../models/model-gru-selfAttention.h5'
use_nnet = model_path.endswith('.h5')
use_time_series  = any(t in model_path for t in ['gru','lstm','rnn','cnn','tcn'])
model = load_model(model_path, custom_objects={'huber':huber_loss(),'MultiHeadAttention':MultiHeadAttention,
      'SeqSelfAttention':SeqSelfAttention,'Lookahead':Lookahead,
      'RAdam':RAdam,'AdaBound':AdaBound}) if(use_nnet) else joblib.load(model_path)

results_dir = '../results/'
for file_name in os.listdir(results_dir):
    if file_name.endswith('.png') or (file_name.startswith('data') and file_name.endswith('.csv')):
        os.unlink(results_dir + file_name)

def print_metrics(sig_orig, dsig_orig, sig_pred, dsig_pred):
		mean_sig_orig = np.mean(sig_orig)
		mean_dsig_orig = np.mean(dsig_orig)
		mean_sig_pred = np.mean(sig_pred)
		mean_dsig_pred = np.mean(dsig_pred)
		mean_sig_orig_diff = np.array([x - mean_sig_orig for x in sig_orig])
		mean_dsig_orig_diff = np.array([x - mean_dsig_orig for x in dsig_orig])
		mean_sig_pred_diff = np.array([x - mean_sig_pred for x in sig_pred])
		mean_dsig_pred_diff = np.array([x - mean_dsig_pred for x in dsig_pred])
       
		rmse_sig = math.sqrt(mean_squared_error(sig_orig, sig_pred ))
		rmse_dsig = math.sqrt(mean_squared_error(dsig_orig,  dsig_pred))
		max_sig = max(abs(sig_orig - sig_pred))
		max_dsig = max(abs(dsig_orig - dsig_pred))   
		min_sig = min(abs(sig_orig - sig_pred))
		min_dsig = min(abs(dsig_orig - dsig_pred))
		rse_sig = rmse_sig/math.sqrt(sum(mean_sig_orig_diff*mean_sig_orig_diff))
		rse_dsig = rmse_dsig/math.sqrt(sum(mean_dsig_orig_diff*mean_dsig_orig_diff))
		corr_sig = (sum(mean_sig_orig_diff*mean_sig_pred_diff))
		corr_sig = corr_sig/math.sqrt(sum(mean_sig_orig_diff*mean_sig_orig_diff)*sum(mean_sig_pred_diff*mean_sig_pred_diff))
		corr_dsig = (sum(mean_dsig_orig_diff*mean_dsig_pred_diff))
		corr_dsig = corr_dsig/math.sqrt(sum(mean_dsig_orig_diff*mean_dsig_orig_diff)*sum(mean_dsig_pred_diff*mean_dsig_pred_diff))    
		print(str(rmse_sig)+','+str(rmse_dsig)+','+str(max_sig)+','+str(max_dsig)+','+str(min_sig)+','+str(min_dsig)
    +','+str(rse_sig)+','+str(rse_dsig)+','+str(corr_sig)+','+str(corr_dsig))

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
	print('data')
	print('rmse(stress), rmse(stress derivative), max_err(stress), max_err(stress derviative), min_err(stress), min_err(stress derivative), rse(stress), rse(stress derviative), corr(stress), corr(stress derivative)') 
	for i in range(0,num_tests):
		indices       = data_noiter.index[data_noiter['testid'] == (i+1)].tolist()
		original_data = np.array(data_noiter)[indices, :]
		pred_data     = np.array(data_scaled_noiter)[indices, :] if(use_nnet) else np.array(data_noiter)[indices, :]
		if(use_time_series):
		    pred_data = pred_data[:, time_series_feature_columns]
		    prediction_tmp = model.predict(InputToTimeSeries(pred_data))
		else:
		    pred_data = pred_data[:, feature_columns]
		    prediction_tmp = model.predict(pred_data)
		if(len(prediction_tmp.shape) == 3):
		    prediction = np.zeros((prediction_tmp.shape[0], prediction_tmp.shape[2]))
		    for itarg in range(0, len(target_columns)):
		        prediction[:, itarg] = prediction_tmp[:, time_series_steps - 1, itarg]
		else:
		    prediction = prediction_tmp
		if(use_nnet):
		    for itarg in range(0, len(target_columns)):
		        prediction[:, itarg] = prediction[:, itarg] * scaler.data_range_[target_columns[itarg]]  + scaler.data_min_[target_columns[itarg]]
		nlen = len(prediction)-time_series_steps
		print_metrics(original_data[0:nlen,target_columns[0]], original_data[0:nlen,target_columns[1]], prediction[0:nlen, 0], prediction[0:nlen, 1])
		df = pd.DataFrame(data = { 'time': original_data[0:nlen,0],
                               'sigma': original_data[0:nlen,target_columns[0]],
                               'delta_sigma': original_data[0:nlen,target_columns[1]],
                               'sigma pred': prediction[0:nlen, 0],
                               'delta_sigma pred': prediction[0:nlen,1]})
		df.to_csv(results_dir + 'data_pred_test' + str(i+1) + '.csv', index=False)


if(writeDynamicResults):
	print('dynamic')
	print('rmse(stress), rmse(stress derivative), max_err(stress), max_err(stress derviative), min_err(stress), min_err(stress derivative), rse(stress), rse(stress derviative), corr(stress), corr(stress derivative)') 
	for i in range(0,num_tests):
	    try:
	        indices       = data_noiter.index[data_noiter['testid'] == (i+1)].tolist()
	        original_data = np.array(data_noiter)[indices, :]
	        prediction = pd.read_csv(results_dir + "surroHuxley"+str(i+1)+".csv", sep='\s*,\s*', engine='python')
	        prediction = np.array(prediction.loc[::4, ['sigma','delta_sigma']])
        	nlen = len(prediction)-time_series_steps
        	print_metrics(original_data[0:nlen,target_columns[0]], original_data[0:nlen,target_columns[1]], prediction[0:nlen, 0], prediction[0:nlen, 1])
        	df = pd.DataFrame(data = { 'time': original_data[0:nlen,0],
                                     'sigma': original_data[0:nlen,target_columns[0]],
                                     'delta_sigma': original_data[0:nlen,target_columns[1]],
                                     'sigma pred': prediction[0:nlen, 0],
                                     'delta_sigma pred': prediction[0:nlen,1]})
	        df.to_csv(results_dir + 'dynamic_pred_test' + str(i+1) + '.csv', index=False)
	    except:
	        print("Error during processing test No. " + str(i+1))

if(showTestData): drawTestData(data_noiter)
drawTestResults()
results_dir = '../results/'
for file_name in os.listdir(results_dir):
    if(file_name.endswith('.csv')):
        os.unlink(results_dir + file_name)
