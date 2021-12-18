#include <cuda.h>
#include <curand.h>
#define N_NODES 4		//length of the hamiltonian cycle in nodes
#define OFFSPRING_FACTOR 2	//the number of children generated from each individual in the population
#define POPULATION_SIZE 2048	//the maximum number of individuals in the population

#define CUDA_CALL(x) do{ if((x) != cudaSuccess){ \
	printf("Error at %s:%d\n", __FILE__, __LINE__); \
	return EXIT_FAILURE;}} while(0) 

#define CURAND_CALL(x) do{ if((x) != CURAND_STATUS_SUCCESS){ \
	printf("Error at %s:%d\n", __FILE__, __LINE__); \
	return EXIT_FAILURE;}} while(0) 
