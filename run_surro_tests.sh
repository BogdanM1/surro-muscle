#!/bin/bash
cd surro-muscle/src/
python3 convert_h5_to_pb.py
cd $HOME/boxieMexie
rm -rf [1-9]*
cp mexie_exe/shared/FEM_MMNoCa.cfg mexie_exe/shared/FEM_MM.cfg
./run_tests.sh
cp mexie_exe/shared/FEM_MMCa.cfg mexie_exe/shared/FEM_MM.cfg
./run_tests.sh ca

num_fem_mpi=`ps -u bogdan|grep FEM_MPI|wc -l`
while [ $num_fem_mpi != 0 ]
do
    sleep 1 
    num_fem_mpi=`ps -u bogdan|grep FEM_MPI|wc -l`
done
./collect_results.sh >../surro-muscle/results/metrics.txt
