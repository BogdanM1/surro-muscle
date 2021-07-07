import random
import os
from shutil import copy2

num_examples = 40
sim_length = 1.0
max_force = 3.08192E-01

f = open('protocol.dat', 'w')
f.write(f'1\n{sim_length} 2')
f.close()

f = open('input_displacement.dat', 'w')
f.write(f'1\n{sim_length} 0')
f.close()

for i in range(num_examples):
    left_f = random.uniform(0.5 * max_force, 1.5 * max_force)
    right_f = random.uniform(0.0, 0.25 * max_force)
    if(i < 5): 
        temp = left_f 
        left_f = right_f 
        right_f = temp

    left_t = random.uniform(0.0, sim_length/2)
    right_t = random.uniform(sim_length/2, sim_length)

    f = open('input_force.dat', 'w')
    f.write(f'3\n0.0 0\n{round(left_t, 4)} {round(left_f, 4)}\n{sim_length} {round(right_f, 4)}')
    f.close()
    os.makedirs('Force/'+str(i+1))
    copy2('Pak.dat', os.path.join('Force', str(i+1)))
    copy2('protocol.dat', os.path.join('Force', str(i+1)))
    copy2('input_force.dat', os.path.join('Force', str(i+1)))
    copy2('input_displacement.dat', os.path.join('Force', str(i+1)))
