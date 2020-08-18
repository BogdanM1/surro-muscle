for i in {1..90}
do

#skip tests with ca if [ $# -eq 0 ]
if [ $# -eq 0 ] && (([ $i -gt 15 ] && [ $i -lt 41 ]) || ([ $i -gt 60 ] && [ $i -lt 86 ])) 
then 
continue 
fi 

# skip tests without ca if [ $# -gt 0 ]
if [ $# -gt 0 ] && ([ $i -le 15 ] || ([ $i -ge 41 ] && [ $i -le 60 ]) || [ $i -ge 86 ]) 
then 
continue 
fi 
 
rm -rf ${i}
cp -r mexie_exe/ ${i}
cd ${i}/FEMSolver/build
cp tests/$i/Pak.dat Pak.dat
cp tests/$i/displs.dat displs.dat
#echo ${i} > simulation_id.txt
screen -dmL env LD_LIBRARY_PATH=$LD_LIBRARY_PATH mpirun -np 1 ./FEM_MPI
cd ../../../
done


