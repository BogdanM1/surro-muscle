#!/bin/bash
cd surro-muscle/src/
#python3 print_surro_conf.py
cp surro_conf.txt $HOME/boxieMexie/mexie_exe/FEMSolver/build/surro_conf.txt
rm ../results/*

cd $HOME/boxieMexie
rm -rf [1-9]*


start=1
end=15

for i in {1..11}
do 
 ./run_tests.sh $start $end
 num_fem_mpi=`ps -u bogdan|grep FEM_MPI|wc -l`
 while [ $num_fem_mpi != 0 ]
 do
    sleep 1 
    num_fem_mpi=`ps -u bogdan|grep FEM_MPI|wc -l`
 done
start=$((start + 15))
end=$((end + 15))
done

./collect_results.sh >../surro-muscle/results/metrics.txt
