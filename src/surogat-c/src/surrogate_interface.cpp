#include "surrogate_interface.hpp"
#include <fstream>

using namespace std;
using std::vector;

#define MAX_LINE_LENGTH 255


int nfeatures, ntargets, ntimesteps;
 
int surro_fem_execution_phase; // 0 - init , 1 - first iteration of the first step, 2 - any other 
int surro_nqpoints; 
TF_Graph* surro_graph;
TF_Session* surro_session;
vector<TF_Output> surro_input_ops;
vector<TF_Output> surro_out_ops;
vector<int64_t> surro_input_dims;
vector<float> surro_input_values;
vector<float> output_values;

double* surro_data_min;  
double* surro_data_range; 
double surro_scale_min, surro_scale_range;
double init_stretch_value, init_act_value, init_stress_value, init_dstress_value; 

void surro_init(int* n_qpoints, char* model_path, char* conf_file)
{
  char line[MAX_LINE_LENGTH];
  FILE*conf_stream = fopen(conf_file, "r");
  fgets(line, MAX_LINE_LENGTH, conf_stream); 
  fscanf(conf_stream, "%d%d%d%lf%lf\n", &nfeatures, &ntargets, &ntimesteps, &surro_scale_min, &surro_scale_range);
  surro_data_min = (double*)malloc(sizeof(double)*(nfeatures + ntargets));
  surro_data_range = (double*)malloc(sizeof(double)*(nfeatures + ntargets));
  fgets(line, MAX_LINE_LENGTH, conf_stream); 
  for(int i=0; i < (nfeatures+ntargets); i++)
    fscanf(conf_stream,"%lf\n",&surro_data_min[i]);
  fgets(line, MAX_LINE_LENGTH, conf_stream); 
  for(int i=0; i < (nfeatures+ntargets); i++)
    fscanf(conf_stream,"%lf\n",&surro_data_range[i]);    
  fclose(conf_stream);

	surro_fem_execution_phase = 0;
	surro_nqpoints = *n_qpoints;
 
	surro_input_dims = {surro_nqpoints, ntimesteps, nfeatures};
	surro_input_values.resize(surro_nqpoints*ntimesteps*nfeatures);
  
  init_act_value = surro_scale_range*(0.0 - surro_data_min[0])/surro_data_range[0] + surro_scale_min;
  init_stretch_value = surro_scale_range*(1.0 - surro_data_min[1])/surro_data_range[1] + surro_scale_min;
  init_stress_value = surro_scale_range*(0.0 - surro_data_min[2])/surro_data_range[2] + surro_scale_min;
  init_dstress_value = surro_scale_range*(0.0 - surro_data_min[3])/surro_data_range[3] + surro_scale_min;
	
	// load surro_graph
	surro_graph = tf_utils::LoadGraph(model_path);
	if(surro_graph == nullptr) 
	{
		std::cout << "Can't load surro_graph" << std::endl;
		return;
	}
	
	// input/output operations
	surro_input_ops = {{TF_GraphOperationByName(surro_graph, "input_layer"), 0}};
	surro_out_ops = {{TF_GraphOperationByName(surro_graph, "output_layer/BiasAdd"), 0}}; 	//BiasAdd //strided_slice_3
	
	// create surro_session 
	surro_session = tf_utils::CreateSession(surro_graph);
	if(surro_session == nullptr) 
	{
		std::cout << "Can't create surro_session" << std::endl;
		return;
	}	

}

void surro_set_values(int * qindex, double* stretch, double* activation, int *fstStepfstIter)
{
  //printf("%d: %.19lf %.19lf\n",*qindex,*stretch, *activation); // Bogdan:stampa
	int vec_index = (*qindex)*ntimesteps*nfeatures + (ntimesteps - 1)*nfeatures;

  surro_input_values[vec_index] = surro_scale_range*(*activation - surro_data_min[0])/(surro_data_range[0]) + surro_scale_min;
	surro_input_values[vec_index + 1] = surro_scale_range*(*stretch - surro_data_min[1])/( surro_data_range[1] ) + surro_scale_min;	

  // vector is filled-in for the first time: previous time steps are set to be the same as current step
	if(*fstStepfstIter == 1)
	{
		int tmp_vec_index = (*qindex)*ntimesteps*nfeatures + (ntimesteps-1)*nfeatures;
		surro_input_values[tmp_vec_index + nfeatures - 2]  = init_stress_value;
		surro_input_values[tmp_vec_index + nfeatures - 1]  = init_dstress_value;		
		
		for(int istep = 0; istep < ntimesteps-1; istep++)
		{
			tmp_vec_index = (*qindex)*ntimesteps*nfeatures + istep*nfeatures;
			surro_input_values[tmp_vec_index] = init_act_value; 
      surro_input_values[tmp_vec_index + 1] = init_stretch_value;
      surro_input_values[tmp_vec_index + 2] = init_stress_value;
      surro_input_values[tmp_vec_index + 3] = init_dstress_value;
		}
	}	 
}

void surro_converged()
{
	// shift everything by one time steps
	int current_step_index, next_step_index, vec_index;
  double stress, dstress;
	for(int iqpoint = 0; iqpoint < surro_nqpoints; iqpoint++)
	{		
		for(int istep = 0; istep < ntimesteps-1; istep++)
			for(int ifeature = 0; ifeature < nfeatures; ifeature++)
			{
				current_step_index = iqpoint*ntimesteps*nfeatures + istep*nfeatures + ifeature;
				next_step_index = iqpoint*ntimesteps*nfeatures + (istep+1)*nfeatures + ifeature;
				surro_input_values[current_step_index] = surro_input_values[next_step_index];
			}
		// set appropriate stress values for the last time step 
		vec_index = iqpoint*ntimesteps*nfeatures + (ntimesteps - 1)*nfeatures;
    stress = output_values[iqpoint * ntargets];
    dstress = output_values[iqpoint * ntargets + 1];
    
    // descale 
    stress = ((stress-surro_scale_min)/surro_scale_range ) *surro_data_range[nfeatures] + surro_data_min[nfeatures];
	  dstress = ((dstress-surro_scale_min)/surro_scale_range ) *surro_data_range[nfeatures+1] + surro_data_min[nfeatures+1];

	  // scale  
    stress =  surro_scale_range*(stress - surro_data_min[nfeatures-2])/surro_data_range[nfeatures-2]  + surro_scale_min;
    dstress =  surro_scale_range*(dstress - surro_data_min[nfeatures-1])/surro_data_range[nfeatures-1] + surro_scale_min; 
        
		surro_input_values[vec_index + 2] = stress;
		surro_input_values[vec_index + 3] = dstress;	    
    
	}
	
}

void surro_predict()
{
	vector<TF_Tensor*> input_tensors = {tf_utils::CreateTensor(TF_FLOAT, surro_input_dims, surro_input_values)};
 	vector<TF_Tensor*> output_tensors = {nullptr};	
 
 /**************** debug *******************
  cout << "input values\n";
   for(int i=0; i< (int)surro_input_values.size(); i+=4)
   {
    if(i && i % (nfeatures*ntimesteps) == 0) printf("\n\n");   
    printf("%.8lf %.8lf %.8lf %.8lf\n",surro_input_values[i], surro_input_values[i+1], surro_input_values[i+2], surro_input_values[i+3]);
   }
   cout << "\n";
 /*******************************************/
 
	tf_utils::RunSession(surro_session, surro_input_ops, input_tensors, surro_out_ops, output_tensors);
	output_values = tf_utils::GetTensorData<float>(output_tensors[0]);
	tf_utils::DeleteTensors(input_tensors);
	tf_utils::DeleteTensors(output_tensors);	
 
 /*********** debug *******************
  cout << "prediction\n";
  for(int i=0; i< (int) output_values.size(); i+=2)
    printf("%.19lf %.19lf\n",output_values[i], output_values[i+1]);
  cout << "\n";
  /***************************************/
}

void surro_get_values(int *qindex, double * stress, double * dstress)
{
  *stress = (output_values[(*qindex) * ntargets] - surro_scale_min)/surro_scale_range;
	*stress =  (*stress)*surro_data_range[nfeatures] + surro_data_min[nfeatures];
	*dstress = (output_values[(*qindex) * ntargets + 1] - surro_scale_min)/surro_scale_range;
  *dstress = (*dstress)*surro_data_range[nfeatures+1] + surro_data_min[nfeatures+1];
  
   // Bogdan:stampa
   // printf("%d: %.19lf %.19lf\n",*qindex,*stress, *dstress);
}

void surro_destroy()
{
	tf_utils::DeleteSession(surro_session);	
	tf_utils::DeleteGraph(surro_graph); 
	surro_input_dims.clear();
	surro_input_values.clear();
	output_values.clear(); 
  free(surro_data_min); 
  free(surro_data_range); 
}


/*
// test 
int main()
{
	surro_input_dims = {1, ntimesteps, nfeatures};
	surro_input_values.resize(1*ntimesteps*nfeatures, .0);
	
	// load surro_graph
	surro_graph = tf_utils::LoadGraph("/home/bogdan/surro-muscle/models/model.pb");
	if(surro_graph == nullptr) 
	{
		std::cout << "Can't load surro_graph" << std::endl;
		return 0;
	}
	// input/output operations
	surro_input_ops = {{TF_GraphOperationByName(surro_graph, "input_layer"), 0}};
	surro_out_ops = {{TF_GraphOperationByName(surro_graph, "output_layer/BiasAdd"), 0}}; 	
	
	// create surro_session 
	surro_session = tf_utils::CreateSession(surro_graph);
	if(surro_session == nullptr) 
	{
		std::cout << "Can't create surro_session" << std::endl;
		return 0;
	}	
 
 surro_input_values = {0., 1., 0, 0,
 0., 1., 0, 0,
 0., 1., 0, 0,
 0., 1., 0, 0,
 0., 1., 0, 0,
 0., 1., 0, 0,
 0., 1., 0, 0,
 0., 1., 0, 0,
 0., 1., 0, 0,
 0., 1., 0, 0,
 0., 1., 0, 0
  };
 
	vector<TF_Tensor*> input_tensors = {tf_utils::CreateTensor(TF_FLOAT, surro_input_dims, surro_input_values)};
 	vector<TF_Tensor*> output_tensors = {nullptr};	
//	vector<TF_Tensor*> output_tensors = {tf_utils::CreateEmptyTensor(TF_FLOAT, {surro_nqpoints, ntargets}, surro_nqpoints*ntargets*sizeof(float))};	
 
 // debug 
  cout << "input values\n";
   for(int ii=0; ii< (int)surro_input_values.size(); ii+=4)
   {
    cout << surro_input_values[ii] << " " << surro_input_values[ii+1] << " " << surro_input_values[ii+2] << " " << surro_input_values[ii+3] <<"\n ";
   }
   cout << "\n";
 //
 
	tf_utils::RunSession(surro_session, surro_input_ops, input_tensors, surro_out_ops, output_tensors);
	output_values = tf_utils::GetTensorData<float>(output_tensors[0]);
	tf_utils::DeleteTensors(input_tensors);
	tf_utils::DeleteTensors(output_tensors);	
 
 // debug
  cout << "prediction\n";
  for(int i=0; i< (int) output_values.size(); i+=2)
  {
    cout << output_values[i] << " " << output_values[i+1] << "\n ";
  }
  cout << "\n";
  // 
}
*/
