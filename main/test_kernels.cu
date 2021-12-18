#include <stdio.h>
#include <curand.h>
#include <cuda.h>
#include "utils.h"
#include "device_utils.h"
#include "kernels.h"
#include "main.h"

#define VERBOSE 1

void test_init();
void test_ngen();

int main(void){
	//test_init();
	test_ngen();
	
	return 0;
}

void test_init(){
	int *pop;
	int *d_pop;
	unsigned int *d_rands;
	
	pop= (int*) malloc(POPULATION_SIZE*N_NODES*sizeof(int));
	cudaMalloc( (void **) &d_pop, POPULATION_SIZE*N_NODES*sizeof(int) );

	cudaMalloc((void **) &d_rands, POPULATION_SIZE*N_NODES*sizeof(unsigned int));
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234);
	curandGenerate(gen, (unsigned int *) d_rands, POPULATION_SIZE*N_NODES*sizeof(unsigned int));

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	
	dim3 threads(32,prop.maxThreadsDim[0]/32,1);
	dim3 blocks(ceil(POPULATION_SIZE/prop.maxThreadsDim[0]),1,1); 
	if(POPULATION_SIZE< prop.maxThreadsDim[0]){ 
		threads.y = POPULATION_SIZE/32;
		blocks.x = 1;
	}	
	

	printf("launching with (%d, %d,%d) threads and %d blocks\n", 
			threads.x, threads.y, threads.z, blocks.x);
	init_pop_s<<<blocks, threads>>>(d_pop, POPULATION_SIZE, N_NODES, d_rands, (POPULATION_SIZE/32)*3);
	
	cudaMemcpy( pop, d_pop, POPULATION_SIZE*N_NODES*sizeof(int), cudaMemcpyDeviceToHost);
	
	for(int t =0; t<POPULATION_SIZE; ++t){
		for(int s =0; s<N_NODES; ++s){
			printf("%d ", pop[t*N_NODES + s]);
		}
		printf("\n");
	}
	free(pop);
	curandDestroyGenerator(gen);
	cudaFree(d_rands);
	cudaFree(d_pop);
}

void test_ngen(){
	
	int *pop, *off;
	int *d_pop, *d_off;

	unsigned int *d_rands, *rands_init;
	
	pop= (int*) malloc(POPULATION_SIZE*N_NODES*sizeof(int));
	off= (int*) malloc(POPULATION_SIZE*N_NODES*OFFSPRING_FACTOR*sizeof(int));
	
	cudaMalloc((void **) &d_off, POPULATION_SIZE*OFFSPRING_FACTOR*N_NODES*sizeof(int));
	cudaMalloc((void **) &d_pop, POPULATION_SIZE*N_NODES*sizeof(int));

	cudaMalloc((void **) &d_rands, ceil(((double)POPULATION_SIZE/32))*3*sizeof(unsigned int));
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234);
	curandGenerate(gen, (unsigned int *) d_rands, ceil((double)POPULATION_SIZE/32)*3*sizeof(unsigned int));
		
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	
	dim3 threads(32,prop.maxThreadsDim[0]/32,1);
	dim3 blocks(ceil(POPULATION_SIZE/prop.maxThreadsDim[0]),1,1); 
	if(POPULATION_SIZE< prop.maxThreadsDim[0]){ 
		threads.y = (POPULATION_SIZE < 32 ? 1 : POPULATION_SIZE/32);
		//printf("%d ahahah %d \n", ceil(POPULATION_SIZE/32), threads.y);
		blocks.x = 1;
	}
	if(POPULATION_SIZE<32 ){
		threads.x = POPULATION_SIZE;
	}

	cudaMalloc((void **) &rands_init, POPULATION_SIZE*N_NODES*sizeof(unsigned int));
	curandGenerate(gen, (unsigned int *) rands_init, POPULATION_SIZE*N_NODES*sizeof(unsigned int));
	
	printf("launching init_pop_s with (%d, %d,%d) threads and %d blocks\n", 
			threads.x, threads.y, threads.z, blocks.x);
	init_pop_s<<<blocks, threads>>>(d_pop, POPULATION_SIZE, N_NODES, rands_init, (POPULATION_SIZE/32)*3);

#if VERBOSE
	cudaMemcpy(pop, d_pop, POPULATION_SIZE*N_NODES*sizeof(int), cudaMemcpyDeviceToHost);
	for(int t =0; t<POPULATION_SIZE; ++t){
		for(int s =0; s<N_NODES; ++s){
			printf("%d ", pop[t*N_NODES + s]);
		}
		printf("\n");
	}
#endif
	cudaDeviceSynchronize();
	printf("launching naive_generation with (%d, %d,%d) threads and %d blocks\n", 
			threads.x, threads.y, threads.z, blocks.x);
	naive_generation<<<blocks, threads>>>( 	d_pop, 
						POPULATION_SIZE, 
						N_NODES, 
						d_off, 
						OFFSPRING_FACTOR, 
						d_rands);
	

	cudaMemcpy(off, d_off, POPULATION_SIZE*N_NODES*OFFSPRING_FACTOR*sizeof(int), cudaMemcpyDeviceToHost);
	
#if VERBOSE
	for(int t =0; t<POPULATION_SIZE*OFFSPRING_FACTOR; ++t){
		for(int s =0; s<N_NODES; ++s){
			printf("%d ", off[t*N_NODES + s]);
		}
		printf("\n");
	}	
#endif	
	free(pop);
	curandDestroyGenerator(gen);
	cudaFree(d_rands);
	cudaFree(d_pop);
	cudaFree(d_off);
	cudaFree(rands_init);
}














