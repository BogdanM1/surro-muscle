for i in {1..15}
do
rm -rf ${i}
cp -r mexie_exe/ ${i}
cd ${i}/FEMSolver/build
cp tests/$i/Pak.dat Pak.dat
echo ${i} > simulation_id.txt
screen -dmL mpirun -np 1 ./FEM_MPI
cd ../../../
done



