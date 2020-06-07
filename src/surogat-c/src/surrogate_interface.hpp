#include "tf_utils.hpp"
#include "scope_guard.hpp"
#include <iostream>
#include <vector>

void surro_init(int* n_qpoints, char* model_path="/home/bogdan/surro-muscle/models/model.pb", char* min_max_file="min_max.txt");
void surro_set_values(int * qindex, double* stretch, double* activation);
void surro_converged();
void surro_predict();
void surro_get_values(int *qindex, double * stress, double * dstress);
void surro_destroy();
