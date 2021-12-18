#include <stdio.h>
#include <curand.h>
#include <cuda.h>
#include <float.h>
#include "utils.h"
#include "device_utils.h"
#include "kernels.h"
#include "main.h"

#define PRINT_SUMMARY 1

int * init_population(curandGenerator_t gen, int n_dim, int population_dim){
	int *pop;
	unsigned int * rands;

	
	cudaMalloc((void **) &pop, n_dim*population_dim*sizeof(unsigned int));
	cudaMalloc((void **) &rands, n_dim*population_dim*sizeof(unsigned int));
	
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

int main(){

	//allocate graph
	float **g = graph_generate(N_NODES);
	print_graph(g, N_NODES);
	float **m= graph_to_mat(g, N_NODES);
	float *vec_graph = mat_to_vec(m, N_NODES);
	float *d_vec_graph;
	cudaMalloc((void **) &d_vec_graph, N_NODES*N_NODES*sizeof(float));
	cudaMemcpy( d_vec_graph,  vec_graph, N_NODES*N_NODES*sizeof(float), cudaMemcpyHostToDevice);

	free(g);
	free(m);

	//allocate data arrays			DIM
	int * d_population; 			//POPULATION_SIZE*N_NODES
	
	
	int *d_offspring;			//POPULATION_SIZE*N_NODES*OFFSPRING_FACTOR

	float *d_fitness;			//POPULATION_SIZE*OFFSPRING_FACTOR
	int *d_auxiliary;			

	unsigned int *d_shuffle_rands;		//POPULATION_SIZE
	
	unsigned int *d_genetic_rands;		//N_WARPS*OFFSPRING_FACTOR*3

	cudaMalloc((void **) &d_offspring, POPULATION_SIZE*N_NODES*OFFSPRING_FACTOR*sizeof(int));
	cudaMalloc((void **) &d_auxiliary, POPULATION_SIZE*OFFSPRING_FACTOR*sizeof(int));
	cudaMalloc((void **) &d_fitness, POPULATION_SIZE*OFFSPRING_FACTOR*sizeof(float));
	cudaMalloc((void **) &d_shuffle_rands, POPULATION_SIZE*sizeof(unsigned int));

	int n_warps = POPULATION_SIZE/32;
	if(POPULATION_SIZE <32){
		n_warps =1;
	}

	cudaMalloc((void **) &d_genetic_rands, n_warps*OFFSPRING_FACTOR*3*sizeof(unsigned int));
		
	
	
	//create curand generator
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234);

	
	//initialize the data
	d_population = init_population(gen, N_NODES, POPULATION_SIZE);

	
	//kernel parameters
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


	//support variables
	int *global_best = (int*) malloc(N_NODES*sizeof(int));
	int *current_best = (int*) malloc(N_NODES*sizeof(int));
	float best_fitness = FLT_MAX;
	float current_fitness;
	float fitnesses[N_ITERATIONS];
	
	
	//main loop
	for(int t=0; t<N_ITERATIONS; ++t){
		//generate random numbers for offspring generation
		curandGenerate(gen, (unsigned int *) d_genetic_rands, n_warps*OFFSPRING_FACTOR*3*sizeof(unsigned int));

		printf("it %d: generating the offspring\n", t);

		init_pop_s<<<blocks, threads>>>(d_population, POPULATION_SIZE, N_NODES, d_genetic_rands);
		
		printf("it %d: applying selection", t);
		naive_selection<<<blocks, threads>>>(	d_offspring,
							d_population,
							N_NODES,
							POPULATION_SIZE,
							OFFSPRING_FACTOR,
							d_vec_graph,
							d_fitness,
							d_auxiliary);

		//record best solution
		cudaMemcpy( current_best, d_population, N_NODES*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy( &current_fitness, d_fitness, sizeof(float), cudaMemcpyDeviceToHost);

		fitnesses[t] = current_fitness;
		if(current_fitness<best_fitness){
			printf("it %d: improvement found!", t);
			for(int s=0; s<N_NODES; ++s){
				global_best[s] = current_best[s];
			}
		} 

		//shuffle
		thrust_shuffle(d_population, d_offspring, d_auxiliary, gen, d_shuffle_rands, N_NODES, POPULATION_SIZE);
				

	}
	
#if PRINT_SUMMARY
	printf("summary of iterations:\n");

	for(int t=0; t<N_ITERATIONS; ++t){
		printf("%.2f ->", fitnesses[t]);
	}
	printf("\n");
#endif
	
	printf("best solution found has path length %.2f\n", best_fitness);

	for(int t=0; t<N_NODES; ++t){
		printf("%d ->", global_best[t]);
	}
	printf("\n");

	free(global_best);
	free(current_best);
	cudaFree(d_population);
	cudaFree(d_offspring);
	cudaFree(d_shuffle_rands);
	cudaFree(d_genetic_rands);
	cudaFree(d_auxiliary);
	

}















