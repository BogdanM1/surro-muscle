cd surro-muscle/src/
python3 convert_h5_to_pb.py
cd $HOME/boxieMexie
./run_tests.sh
while($(ps -u bogdan|grep FEM_MPI|wc -l) -gt 0); do sleep 1; done
./collect_results.sh >../surro-muscle/results/metrics.txt
