#include "kernels.cu"
#include "generation_kernels.cu"
#include "selection_kernels.cu"

#ifndef DEVICEUTILS_H
#include "device_utils.h"
#endif

__global__ void init_pop(int *pop, int pop_dim, int n_dim, unsigned int *random_nums, int r_dim);
__global__ void naive_generation(int* population, 
				int population_dim, 
				int n_dim, int *offspring,
				int offspring_factor,
				unsigned int *random_nums);

