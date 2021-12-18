#include <stdio.h>
#include <curand.h>
#include <cuda.h>
#include "kernels.h"

void print_pop(int *d_pop, int pop_dim){
	int *pop = (int *) malloc(pop_dim *N_NODES* sizeof(int));
	cudaMemcpy( pop, d_pop, pop_dim*N_NODES*sizeof(int), cudaMemcpyDeviceToHost);
	for(int s=0; s< pop_dim; ++s){
		for(int k =0 ; k<N_NODES; ++k){			
			printf("%d ", pop[s*N_NODES + k]);
		}
		printf("\n");
	}

	free(pop);
}

void print_popfit(int *d_pop,float * d_fit, int pop_dim){
	int *pop = (int *) malloc(pop_dim *N_NODES* sizeof(int));
	float * fit = (float *) malloc(pop_dim * sizeof(float));
	cudaMemcpy( pop, d_pop, pop_dim*N_NODES*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy( fit, d_fit, POPULATION_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	for(int s=0; s< pop_dim; ++s){
		printf("%d) ", s);
		for(int k =0 ; k<N_NODES; ++k){			
			printf("%d ", pop[s*N_NODES + k]);
		}
		printf("- %.2f\n", fit[s]);
	}

	free(pop);
}

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
		threads.y = 1;
        }

	
	
	init_pop_s<<<blocks, threads>>>(pop, population_dim, n_dim, rands);
	cudaDeviceSynchronize();
	
	
	cudaFree(rands);
	return cudaSuccess;

}

