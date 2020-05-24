#include "surrogate_interface.hpp"
using namespace std;
using std::vector;

#define nfeatures 4
#define ntargets 2
#define ntimesteps 11
 
 
int fem_execution_phase; // 0 - init , 1 - first iteration of the first step, 2 - any other 
int nqpoints; 
TF_Graph* graph;
TF_Session* session;
vector<TF_Output> input_ops;
vector<TF_Output> out_ops;
vector<int64_t> input_dims;
vector<double> input_values, output_values;


void init(int* n_qpoints, char* model_path)
{
	fem_execution_phase = 0;
	nqpoints = *n_qpoints;
	input_dims = {nqpoints, ntimesteps, nfeatures};
	input_values.resize(nqpoints*ntimesteps*nfeatures, .0);
	output_values.resize(nqpoints*ntargets, .0);
	
	// load graph
	graph = tf_utils::LoadGraph(model_path);
	if(graph == nullptr) 
	{
		std::cout << "Can't load graph" << std::endl;
		return;
	}
	
	// input/output operations
	input_ops = {{TF_GraphOperationByName(graph, "input_layer"), 0}};
	out_ops = {{TF_GraphOperationByName(graph, "output_layer/BiasAdd"), 0}}; 	
	
	// create session 
	session = tf_utils::CreateSession(graph);
	if(session == nullptr) 
	{
		std::cout << "Can't create session" << std::endl;
		return;
	}	
}

void set_values(int * qindex, double* stretch, double* activation)
{
	int vec_index = (*qindex)*ntimesteps*nfeatures + (ntimesteps - 1)*nfeatures;
	input_values[vec_index] = *stretch;
	input_values[vec_index + 1] = *activation;	
	input_values[vec_index + 2] = output_values[(*qindex) * ntargets];
	input_values[vec_index + 3] = output_values[(*qindex) * ntargets + 1];
	
	// vector is filled-in for the first time: previous time steps are set to be the same as current step
	if(fem_execution_phase == 1)
	{
		int tmp_vec_index; 
		for(int istep = 0; istep < ntimesteps-1; istep++)
		{
			tmp_vec_index = (*qindex)*ntimesteps*nfeatures + istep*nfeatures;
			for(int ifeature = 0; ifeature < nfeatures; ifeature++)
				input_values[tmp_vec_index + ifeature]  = input_values[vec_index + ifeature]; 
		}
	}	
	
}

void converged()
{
	if(fem_execution_phase < 2)
	{
		fem_execution_phase++;
		return; 
	}
	// shift everything by one time steps
	int current_step_index, next_step_index, vec_index;
	for(int iqpoint = 0; iqpoint < nqpoints; iqpoint++)
	{
		// set appropriate stress values for the last time step 
		vec_index = iqpoint*ntimesteps*nfeatures + (ntimesteps - 1)*nfeatures;
		input_values[vec_index + 2] = output_values[iqpoint * ntargets];
		input_values[vec_index + 3] = output_values[iqpoint * ntargets + 1];	
		
		for(int istep = 0; istep < ntimesteps-1; istep++)
			for(int ifeature = 0; ifeature < nfeatures; ifeature++)
			{
				current_step_index = iqpoint*ntimesteps*nfeatures + istep*nfeatures + ifeature;
				next_step_index = iqpoint*ntimesteps*nfeatures + (istep+1)*nfeatures + ifeature;
				input_values[current_step_index] = input_values[next_step_index];
			}
	}
	
}

void predict()
{
	vector<TF_Tensor*> input_tensors = {tf_utils::CreateTensor(TF_FLOAT, input_dims, input_values)};
	vector<TF_Tensor*> output_tensors = {nullptr};	
	tf_utils::RunSession(session, input_ops, input_tensors, out_ops, output_tensors);
	output_values = tf_utils::GetTensorData<double>(output_tensors[0]);
	tf_utils::DeleteTensors(input_tensors);
	tf_utils::DeleteTensors(output_tensors);	
}

void get_values(int *qindex, double * stress, double * dstress)
{
	*stress = output_values[(*qindex) * ntargets];
	*dstress = output_values[(*qindex) * ntargets + 1];
}

void destroy()
{
	tf_utils::DeleteSession(session);	
	tf_utils::DeleteGraph(graph); 
	input_dims.clear();
	input_values.clear();
	output_values.clear(); 
}


int main() {
  int np = 4;
  double a=.1, l=1, stress, dstress;
  int qi=0;
    
  init(&np);
  for(int istep = 0; istep < 3; istep++)
  {
    converged();
    a +=  .05*((double) rand() / (RAND_MAX))*a;
    for(int iter = 0; iter < 5; iter++)
    {
      l -=  .05*((double) rand() / (RAND_MAX))*l;
      for(int qi = 0; qi < 4; qi++)
      {
        set_values(&qi, &l, &a);
        predict();
        get_values(&qi, &stress, &dstress);
        std::cout << stress << " " << dstress << std::endl;
      }
    }
  }
  
  
  destroy();
  return 0;
}
