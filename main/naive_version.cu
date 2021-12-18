#include <stdio.h>
#include <curand.h>
#include <cuda.h>
#include "utils.h"
#include "device_utils.h"
#include "kernels.h"
#include "main.h"

int * init_population(curandGenerator_t gen, int n_dim, int population_dim){
	int *pop;
	unsigned int * rands;

	
	cudaMalloc((void **) &pop, n_dim*population_dim*sizeof(unsigned int));
	cudaMalloc((void **) &rands, n_dim*population_dim*n_dim*sizeof(unsigned int));
	
	curandGenerate(gen, (unsigned int *) rands, population_dim*n_dim*sizeof(unsigned int));
	
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	
	dim3 threads(32,prop.maxThreadsDim[0]/32,1);
	dim3 blocks(ceil(population_dim/prop.maxThreadsDim[0]),1,1); 
	if(population_dim< prop.maxThreadsDim[0]){ 
		threads.y = (population_dim < 32 ? 1 : population_dim/32);
		blocks.x = 1;
	}
	if(population_dim<32 ){
		threads.x = population_dim;
	}

	init_pop_s<<<blocks, threads>>>(pop, population_dim, n_dim, rands);
	
	cudaFree(rands);
	return pop;

}

