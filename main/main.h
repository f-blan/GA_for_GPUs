#include <cuda.h>
#include <curand.h>

#define MAIN_H

#define N_NODES 48		//length of the hamiltonian cycle in nodes. Def: 60
#define OFFSPRING_FACTOR 2	//the number of children generated from each individual in the population. Def: 3
#define POPULATION_SIZE 1024	//the maximum number of individuals in the population. Def: 1024
#define N_ITERATIONS 500	//number of iterations of the algorithm. Def: 500
#define THREADS_PER_BLOCK 64	//maximum number of threads per block. Def: 64

#define COMPILE_SHARED 1	//static shared memory size gives compilation errors even when shared memory is not used
#define USE_ISLAND_SELECTION 0
#define USE_ISLAND_GENERATION 0


#define CUDA_CALL(x) do{ if((x) != cudaSuccess){ \
	printf("Error at %s:%d\n", __FILE__, __LINE__); \
	return EXIT_FAILURE;}} while(0) 

#define CURAND_CALL(x) do{ if((x) != CURAND_STATUS_SUCCESS){ \
	printf("Error at %s:%d\n", __FILE__, __LINE__); \
	return EXIT_FAILURE;}} while(0) 




