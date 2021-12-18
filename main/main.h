#include <cuda.h>
#include <curand.h>

#define MAIN_H

#define N_NODES 60		//length of the hamiltonian cycle in nodes
#define OFFSPRING_FACTOR 3	//the number of children generated from each individual in the population
#define POPULATION_SIZE 1024	//the maximum number of individuals in the population
#define N_ITERATIONS 500	//number of iterations of the algorithm
#define THREADS_PER_BLOCK 64	//maximum number of threads per block

#define CUDA_CALL(x) do{ if((x) != cudaSuccess){ \
	printf("Error at %s:%d\n", __FILE__, __LINE__); \
	return EXIT_FAILURE;}} while(0) 

#define CURAND_CALL(x) do{ if((x) != CURAND_STATUS_SUCCESS){ \
	printf("Error at %s:%d\n", __FILE__, __LINE__); \
	return EXIT_FAILURE;}} while(0) 




