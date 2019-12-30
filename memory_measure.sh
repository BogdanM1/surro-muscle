processid=`ps -u msvicevic | grep FEM_MPI | awk '{print $1}'`
while true; do pmap $processid | tail -n 1 | awk '/[0-9]K/{print $2}'; sleep 0.1; done
