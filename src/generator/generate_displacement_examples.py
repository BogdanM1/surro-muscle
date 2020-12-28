import random
import os
from shutil import copy2

num_examples = 20
sim_length = 1.0

f = open('protocol.dat', 'w')
f.write(f'1\n{sim_length} 1')
f.close()

f = open('input_force.dat', 'w')
f.write(f'1\n{sim_length} 0')
f.close()

for i in range(num_examples):
    left_d = random.uniform(-0.5, 0.0)
    right_d = random.uniform(0.0, 0.5)

    left_t = random.uniform(0.0, sim_length/2)
    right_t = random.uniform(sim_length/2, sim_length)

    f = open('input_displacement.dat', 'w')
    f.write(f'3\n0.0 0\n{round(left_t, 4)} {round(left_d, 4)}\n{sim_length} {round(right_d, 4)}')
    f.close()
    os.mkdir('Displacements/Example'+str(i+1))
    copy2('protocol.dat', os.path.join('Displacements', 'Example'+str(i+1)))
    copy2('input_force.dat', os.path.join('Displacements', 'Example'+str(i+1)))
    copy2('input_displacement.dat', os.path.join('Displacements', 'Example'+str(i+1)))
