cd $HOME
cd surro-muscle/src
python3 convert_h5_to_pb.py
python3 print_surro_conf.py
cd $HOME
scp surro-muscle/models/model.h5 bogdan@147.91.200.54:/home/bogdan/surro-muscle/models
scp surro-muscle/models/model.pb bogdan@147.91.200.54:/home/bogdan/surro-muscle/models
scp surro-muscle/src/*.* bogdan@147.91.200.54:/home/bogdan/surro-muscle/src
scp surro-muscle/src/surogat-c/src/*.* bogdan@147.91.200.54:/home/bogdan/surro-muscle/src/surogat-c/src
