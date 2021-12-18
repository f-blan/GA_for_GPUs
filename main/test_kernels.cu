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
void test_sel();
void test_shuffle();

int main(void){
	//test_init();
	test_ngen();
	//test_sel();
	//test_shuffle();	
	

	return 0;
}

void test_shuffle(){
	int *pop, *haux;
	int *d_pop,*d_off, *aux,*out;
	unsigned int *d_rands, *d_srands;
	
	pop= (int*) malloc(POPULATION_SIZE*N_NODES*sizeof(int));
	haux= (int*) malloc(POPULATION_SIZE*sizeof(int));
	cudaMalloc( (void **) &d_pop, POPULATION_SIZE*N_NODES*sizeof(int) );
	cudaMalloc( (void **) &d_off, POPULATION_SIZE*N_NODES*sizeof(int) );
	cudaMalloc( (void **) &out, POPULATION_SIZE*N_NODES*sizeof(int) );
	cudaMalloc( (void **) &aux, POPULATION_SIZE*sizeof(int) );

	cudaMalloc((void **) &d_rands, POPULATION_SIZE*N_NODES*sizeof(unsigned int));
	cudaMalloc((void **) &d_srands, POPULATION_SIZE*sizeof(unsigned int));
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234);
	curandGenerate(gen, (unsigned int *) d_rands, POPULATION_SIZE*N_NODES*sizeof(unsigned int));
	curandGenerate(gen, (unsigned int *) d_srands, POPULATION_SIZE*sizeof(unsigned int));
	
	for(int t=0; t<POPULATION_SIZE; ++t) haux[t] = t;

	cudaMemcpy( aux, haux, POPULATION_SIZE*sizeof(int), cudaMemcpyHostToDevice);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	
	dim3 threads(32,prop.maxThreadsDim[0]/32,1);
	dim3 blocks(ceil(POPULATION_SIZE/prop.maxThreadsDim[0]),1,1); 
	if(POPULATION_SIZE< prop.maxThreadsDim[0]){ 
		threads.y = (POPULATION_SIZE < 32 ? 1 : POPULATION_SIZE/32);
		blocks.x = 1;
	}
	if(POPULATION_SIZE<32 ){
		threads.x = POPULATION_SIZE;
	}
	
	printf("launching offspring init with (%d, %d,%d) threads and %d blocks\n", 
			threads.x, threads.y, threads.z, blocks.x);
	init_pop_s<<<blocks, threads>>>(	d_pop, POPULATION_SIZE,
						 	N_NODES, d_rands);
#if VERBOSE	
	cudaMemcpy( pop, d_pop, POPULATION_SIZE*N_NODES*sizeof(int), cudaMemcpyDeviceToHost);
	
	for(int t =0; t<POPULATION_SIZE; ++t){
		for(int s =0; s<N_NODES; ++s){
			printf("%d ", pop[t*N_NODES + s]);
		}
		printf("\n");
	}
#endif
	printf("launching shuffle init with (%d, %d,%d) threads and %d blocks\n", 
			threads.x, threads.y, threads.z, blocks.x);
	thrust_shuffle(d_pop, d_off, aux, gen, d_srands, N_NODES, POPULATION_SIZE);
#if VERBOSE	
	cudaMemcpy( pop, d_pop, POPULATION_SIZE*N_NODES*sizeof(int), cudaMemcpyDeviceToHost);
	
	for(int t =0; t<POPULATION_SIZE; ++t){
		for(int s =0; s<N_NODES; ++s){
			printf("%d ", pop[t*N_NODES + s]);
		}
		printf("\n");
	}
#endif
	

	free(pop);
	cudaFree(aux);
	curandDestroyGenerator(gen);
	cudaFree(d_rands);
	cudaFree(d_pop);
	cudaFree(d_off);
	cudaFree(d_srands);
}

void test_sel(){
	int *pop, *off;
	int *d_pop, *d_off, *aux;
	float *d_eval;
	unsigned int *d_rands;

	pop= (int*) malloc(POPULATION_SIZE*N_NODES*sizeof(int));
	off = (int*) malloc(POPULATION_SIZE*N_NODES*OFFSPRING_FACTOR*sizeof(int));

	cudaMalloc((void **) &d_off, POPULATION_SIZE*OFFSPRING_FACTOR*N_NODES*sizeof(int));
	cudaMalloc((void **) &d_eval, POPULATION_SIZE*OFFSPRING_FACTOR*N_NODES*sizeof(float));
	cudaMalloc((void **) &aux, POPULATION_SIZE*OFFSPRING_FACTOR*sizeof(int));
	cudaMalloc((void **) &d_pop, POPULATION_SIZE*N_NODES*sizeof(int));

	cudaMalloc((void **) &d_rands, POPULATION_SIZE*OFFSPRING_FACTOR*N_NODES*sizeof(unsigned int));

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234);
	curandGenerate(gen, (unsigned int *) d_rands, POPULATION_SIZE*OFFSPRING_FACTOR*N_NODES*sizeof(unsigned int));

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);

	dim3 threads_sel(32,prop.maxThreadsDim[0]/32,1);
	dim3 blocks_sel(ceil((POPULATION_SIZE*OFFSPRING_FACTOR)/prop.maxThreadsDim[0]),1,1); 
	if(POPULATION_SIZE< prop.maxThreadsDim[0]){ 
		threads_sel.y = ((POPULATION_SIZE*OFFSPRING_FACTOR) < 32 ? 1 : (POPULATION_SIZE*OFFSPRING_FACTOR)/32);
		//printf("%d ahahah %d \n", ceil(POPULATION_SIZE/32), threads.y);
		blocks_sel.x = 1;
	}
	if(POPULATION_SIZE*OFFSPRING_FACTOR<32 ){
		threads_sel.x = (POPULATION_SIZE*OFFSPRING_FACTOR);
	}
	
	printf("launching offspring init with (%d, %d,%d) threads and %d blocks\n", 
			threads_sel.x, threads_sel.y, threads_sel.z, blocks_sel.x);
	init_pop_s<<<blocks_sel, threads_sel>>>(	d_off, POPULATION_SIZE*OFFSPRING_FACTOR,
						 	N_NODES, d_rands);
	
#if VERBOSE
	cudaMemcpy(off, d_off, POPULATION_SIZE*OFFSPRING_FACTOR*N_NODES*sizeof(int), cudaMemcpyDeviceToHost);
	for(int t =0; t<POPULATION_SIZE*OFFSPRING_FACTOR; ++t){
		for(int s =0; s<N_NODES; ++s){
			printf("%d ", off[t*N_NODES + s]);
		}
		printf("\n");
	}
#endif
	float **g = graph_generate(N_NODES);
	float **mat = graph_to_mat(g, N_NODES);
	float *v = mat_to_vec(mat, N_NODES); 
	float *d_v;

	
	cudaMalloc((void **) &d_v, N_NODES*N_NODES*sizeof(float));
	cudaMemcpy(d_v, v, N_NODES*N_NODES*sizeof(float), cudaMemcpyHostToDevice);
#if VERBOSE
	print_mat(mat, N_NODES);
	for(int t=0; t<N_NODES*N_NODES;++t){
		printf("%.1f ", v[t]);
	}
	printf("\n");
#endif
	printf("launching naive_selection with (%d, %d,%d) threads and %d blocks\n", 
			threads_sel.x, threads_sel.y, threads_sel.z, blocks_sel.x);
	naive_selection<<<blocks_sel, threads_sel>>>(	d_off,
							d_pop,
							N_NODES,
							POPULATION_SIZE,
							OFFSPRING_FACTOR,
							d_v,
							d_eval,
							aux);
	
#if VERBOSE
	
	cudaMemcpy( pop, d_pop, POPULATION_SIZE*N_NODES*sizeof(int), cudaMemcpyDeviceToHost);
	for(int t =0; t<POPULATION_SIZE; ++t){
		for(int s =0; s<N_NODES; ++s){
			printf("%d ", pop[t*N_NODES + s]);
		}
		printf("\n");
	}
#endif
	free(pop);
	curandDestroyGenerator(gen);
	cudaFree(d_rands);
	cudaFree(d_pop);
	cudaFree(d_off);
	free(g);
	free(mat);
	free(v);
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
	init_pop_s<<<blocks, threads>>>(d_pop, POPULATION_SIZE, N_NODES, d_rands);
	
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
	init_pop_s<<<blocks, threads>>>(d_pop, POPULATION_SIZE, N_NODES, rands_init);

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














