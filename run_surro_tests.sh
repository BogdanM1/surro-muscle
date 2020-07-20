#!/bin/bash
cd surro-muscle/src/
python3 convert_h5_to_pb.py
cd $HOME/boxieMexie
./run_tests.sh

num_fem_mpi=`ps -u bogdan|grep FEM_MPI|wc -l`
while [ $num_fem_mpi != 0 ]
do
    sleep 1 
    num_fem_mpi=`ps -u bogdan|grep FEM_MPI|wc -l`
done
./collect_results.sh >../surro-muscle/results/metrics.txt
