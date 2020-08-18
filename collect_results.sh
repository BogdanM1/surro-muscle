for i in {1..90}
do
cp ${i}/FEMSolver/build/surroHuxley.csv  $HOME/surro-muscle/results/surroHuxley${i}.csv
#cp ${i}/FEMSolver/build/postHUXLEYFI.csv  $HOME/surro-muscle/data/tests/test${i}.csv
done


cd ../surro-muscle/src
python3 postprocess.py


