#include "tf_utils.hpp"
#include "scope_guard.hpp"
#include <iostream>
#include <vector>

void init(int* n_qpoints, char* model_path="../../../models/model.pb");
void set_values(int * qindex, double* stretch, double* activation);
void converged();
void predict();
void get_values(int *qindex, double * stress, double * dstress);
void destroy();
