# surro-muscle

Data-driven surrogate model of the Huxley muscle model based on Gated Recurrent Unit (GRU), Temporal Convolutional Network (TCN), Nested Long Short-Term Unit (Nested LSTM)..

## /data 
  collected data from finite element simulations with original Huxley muscle model 
  
## /src

   **preprocessing.R** is used to merge all numerical experiments and organize the data into the dataMexie.csv files which are already provided in the data directory.  
   **initialize.py** is used to convert data to time series and initialize variables for training
   ** *Train.py ** files are used to create the models and train them
   **convert_h5_to_pb.py** is used to convert saved model to pb file which can be loaded into C++ code. 
   **postprocessing.py** is used to compare original and surrogate model and draw comparison diagrams 
   **surogat-c** directory contains FEM-surrogate interface 
   **generator** directory contains the numerical experiments generator code, it's used to generate (1) isotonic contraction, (2) quick release, (3) prescribed displacements, (4)  prescribed forces experiments  
    

    
## /models 
  
  saved models
  
  
  
