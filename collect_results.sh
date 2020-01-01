for i in {1..15}
do
cp ${i}/FEMSolver/build/surroHuxley.csv  $HOME/surro-muscle/results/surroHuxley${i}.csv
#cp ${i}/FEMSolver/build/postHUXLEYFI.csv  $HOME/surro-muscle/data/tests/test${i}.csv
done


cd ../surro-muscle/src

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/lustre/home/msvicevic/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/lustre/home/msvicevic/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/lustre/home/msvicevic/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/lustre/home/msvicevic/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate bogdan
python postprocess.py


