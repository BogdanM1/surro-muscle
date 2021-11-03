for (( i=$1; i<=$2; i++ ))
do
  rm -rf ${i}
  cp -r mexie_exe/ ${i}
  
  # tests without ca 
  if [ $i -le 20 ]
  then 
    cp mexie_exe/shared/FEM_MMNoCa.cfg $i/shared/FEM_MM.cfg
  fi  
  
  # quick release tests 
  if [ $i -ge 46 ] && [ $i -le 65 ]
  then 
    cp mexie_exe/shared/FEM_MMNoCa.cfg $i/shared/FEM_MM.cfg
  fi    
  
  # tests with ca 
  if [ $i -ge 21 ] && [ $i -le 45 ] 
  then 
    cp mexie_exe/shared/FEM_MMCa.cfg $i/shared/FEM_MM.cfg
  fi 
  
  # tests with ca and displacements/force
  if [ $i -ge 66 ] && [ $i -le 165 ]
  then
    cp mexie_exe/shared/FEM_MMCa.cfg $i/shared/FEM_MM.cfg   
  fi 
  


  cd ${i}/FEMSolver/build
  cp tests/$i/*.dat .
  screen -dmL env LD_LIBRARY_PATH=$LD_LIBRARY_PATH mpirun -np 1 ./FEM_MPI
  cd ../../../
done


