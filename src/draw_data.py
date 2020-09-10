import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

commands = open("loadData.py").read()
exec(commands)
num_tests = 90
results_dir = '../results/'
ca_dir = '../../boxieMexie/mexie_exe/FEMSolver/build/tests/'

def drawGraph(x, y, name, unit, testid):
    global  results_dir
    plt.figure(figsize=(5, 4), dpi=300)
    plt.plot(x, y, linewidth=3.0, color='rebeccapurple')
    plt.xlabel('Time [s]')
    plt.xlim(left=0)	
    plt.ylabel(name + ' ' + unit)
    plt.ylim(bottom=0)	
    plt.title('Test ' + str(testid) + ' - ' + name)
    plt.tight_layout()
    plt.savefig(results_dir + name + str(testid) + '.png')
    plt.close()

def drawTestData(data):
    for i in range(16, num_tests + 1):
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
        
drawTestData(data_noiter)    
for i in range(16, num_tests + 1):
  filepath = ca_dir + str(i) + '/Pak.dat'
  fp = open(filepath)
  line = fp.readline()
  while line:
    if(line.startswith('C Activation function')): break
    line = fp.readline()
  fp.readline()
  npoints = int(fp.readline())
  fp.readline()
  time = []
  ca = []
  for icnt in range(npoints):
    res = [float(s) for s in fp.readline().split()] 
    time.append(res[0])
    ca.append(res[1])
  fp.close()
  drawGraph(time, ca, 'Ca', '[$\mu$M]',i)
