import math
import os
import sys
import shutil

def create_file(filename, dt = 0.01, points_count = 100, camin = 0, camax = 1, tcmax = 0.01):
    template = open('template/Pak.dat','r')
    outfile = open(filename,'w')
    for line in template:
        if(line.strip() != 'C ovde zameni'):
            outfile.write(line)
        else:
            print('C Activation function', file = outfile)
            print('C Point count', file=outfile)
            print('{:7d}'.format(points_count), file = outfile)
            print('C Time, Activation', file=outfile)
            time = 0.0
            for i in range(points_count+1):
                # hunter Ca equation
                caconc = camin + (camax - camin)*(time/tcmax)*math.exp(1-time/tcmax)
                print('{:.2f}'.format(time),' ', '{:.9f}'.format(caconc), file = outfile)
                time = time + dt
    outfile.close()
    template.close()
    return

shutil.rmtree('samples/')
os.mkdir('samples/')
for i in range(1,101):
    os.mkdir('samples/'+str(i+15))
    create_file('samples/'+str(i+15)+'/Pak.dat', tcmax=i*0.01)
