import random
import os
from shutil import copy2

num_examples = 20
sim_length = 2.0
max_force = 3.08192E-01

f = open('input_displacement.dat', 'w')
f.write(f'2\n0.0 0\n{sim_length} 0.0')
f.close()

for i in range(num_examples):
    left_t = random.uniform(0.7, 0.9 * sim_length)
    force = random.uniform(0.0, 0.25 * max_force)
    f = open('protocol.dat', 'w')
    f.write(f'2\n{round(left_t, 4)} 1\n{sim_length} 2')
    f.close()

    f = open('input_force.dat', 'w')
    if i < 5:
        f.write(f'2\n0.0 0\n{sim_length} {0}')
    else:
        f.write(f'2\n0.0 {round(force, 4)}\n{sim_length} {round(force, 4)}')

    f.close()
    os.mkdir('QR/Example'+str(i+1))
    copy2('protocol.dat', os.path.join('QR', 'Example'+str(i+1)))
    copy2('input_force.dat', os.path.join('QR', 'Example'+str(i+1)))
    copy2('input_displacement.dat', os.path.join('QR', 'Example'+str(i+1)))
