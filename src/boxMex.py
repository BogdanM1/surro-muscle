import numpy as np
import subprocess
import time
from threading import Thread
import os
from flask import Flask
from flask import request
import pandas as pd

app = Flask(__name__)
data = pd.read_csv("../data/dataMexie.csv", header = 0)

# kopirati pak.dat
@app.route('/runTest', methods=['POST'])
def runTest():
	params = request.form.to_dict()

	testid = int(params['testid'])
	guid   = params['guid']

 
    # copy mexie executable
	bashCommand  = "rm -rf ../"+guid
	os.system(bashCommand)	
	mexie_exe = "../mexie_exe/"
	bashCommand  = "cp -r " + mexie_exe + " ../"+guid
	os.system(bashCommand)

    # copy appropriate input files
	copy_from   =  mexie_exe + "FEMSolver/build/tests/" + str(testid) + "/Pak.dat"
	copy_to     = " ../" + guid + "/FEMSolver/build/Pak.dat"
	bashCommand = "cp " + copy_from + " " + copy_to
	os.system(bashCommand)

	mexie_wdir = "../" + guid + "/FEMSolver/build/"

	# set neural network name
	bashCommand = "echo \"" + guid + "\" > " + mexie_wdir + "network.txt"
	os.system(bashCommand)

 	# run simulation
	bashCommand = "mpirun -np 1 ./FEM_MPI"
	FNULL = open(os.devnull, 'w')
	process = subprocess.Popen(bashCommand, cwd = mexie_wdir, shell=True, stdout = FNULL, stderr = FNULL)
	process.wait()

	# error between original and predicted values
	total_err = 0
	data_test = data.query("testid==" + str(testid), inplace = False)
	sigma_orig = np.array(data_test.iloc[:,6])
	dsigma_orig = np.array(data_test.iloc[:,7])

	data_predicted = pd.read_csv(mexie_wdir + "surroHuxley.csv", header=0)
	sigma_predicted = np.array(data_predicted.iloc[::4,8])
	dsigma_predicted = np.array(data_predicted.iloc[::4,10])

	sigma_err  = np.sum(abs(sigma_orig-sigma_predicted))/len(sigma_orig)
	dsigma_err = np.sum(abs(dsigma_orig-dsigma_predicted))/len(dsigma_orig)
	total_err  = sigma_err + dsigma_err 
	return str(total_err)

if __name__ == '__main__':
   app.run(port = 8000, host = "medflow.bioirc.ac.rs", debug = True)
