from random import seed
from random import random
import numpy as np
import math
import os
import sys
import shutil

tmin = 0
tmax = 0.5
tmaxg = 2

disp_min = 0.2
disp_max = 0.5

ngens = 120

def generate_disp(disp_min, disp_max, tmin, tmax, tmaxg):
    t = tmin + random()*(tmax - tmin)
    disp = disp_min + random()*(disp_max - disp_min)
    if(t < (tmin + 0.1) or t > 0.9*tmax):
        time = [tmin, tmaxg]
        displacements = [disp, disp_min] if(t < (tmin + 0.1)) else [disp_min, disp]
    else:
         time = [tmin, t, tmaxg]
         displacements = [disp_min, disp, disp_min]
    return [time, displacements]

def create_file(filename, time, displacements):
    template = open('template/displs.dat','r')
    outfile = open(filename,'w')
    for line in template:
        if(line.strip() != 'C ovde zameni'):
            outfile.write(line)
        else:
            nlen = len(time)
            print(nlen, file = outfile)
            for i in range(nlen):
                print('{:.4f}'.format(time[i]),' {:.9f}'.format(displacements[i]), file = outfile)
    outfile.close()
    template.close()
    return

for i in range(ngens):
    time, displacements = generate_disp(disp_min, disp_max, tmin, tmax, tmaxg)
    os.mkdir('samples/'+str(i+121))
    create_file('samples/'+str(i+121)+'/displs.dat', time, displacements)