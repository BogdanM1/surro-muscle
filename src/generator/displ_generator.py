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

disp_mid_min = -0.2
disp_mid_max = -0.5

disp_end_min = 0.2
disp_end_max = 0.5

ngens = 45

def generate_disp(disp_mid_min, disp_mid_max, disp_end_min, disp_end_max, tmin, tmax, tmaxg):
    t = tmin + random()*(tmax - tmin)
    disp_mid = disp_mid_min + random()*(disp_mid_max - disp_mid_min)
    disp_end = disp_end_min + random()*(disp_end_max - disp_end_min)
    time = [tmin, t, tmaxg]
    displacements = [0, disp_mid, disp_end]
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
    time, displacements = generate_disp(disp_mid_min, disp_mid_max, disp_end_min, disp_end_max, tmin, tmax, tmaxg)
    os.mkdir('samples/'+str(i+ngens+1))
    create_file('samples/'+str(i+ngens+1)+'/displs.dat', time, displacements)
