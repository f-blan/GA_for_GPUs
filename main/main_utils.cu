#include <stdio.h>
#include <curand.h>
#include <cuda.h>
#include "kernels.h"

int init_population(int * pop,curandGenerator_t gen, int n_dim, int population_dim){
	
	unsigned int * rands;

	
	cudaMalloc((void **) &rands, population_dim*n_dim*sizeof(unsigned int));
	
	curandGenerate(gen, (unsigned int *) rands, population_dim*n_dim*sizeof(unsigned int));
	
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	
 
        dim3 threads(32,prop.maxThreadsDim[0]/32,1);
	dim3 blocks(ceil(population_dim/prop.maxThreadsDim[0]),1,1); 
	if(population_dim< prop.maxThreadsDim[0]){ 
		threads.y = population_dim/32;
		blocks.x = 1;
	}
        if(population_dim<32 ){
                threads.x = population_dim;
        }

	
	
	init_pop_s<<<blocks, threads>>>(pop, population_dim, n_dim, rands);
	cudaDeviceSynchronize();
	
	
	cudaFree(rands);
	return cudaSuccess;

}

