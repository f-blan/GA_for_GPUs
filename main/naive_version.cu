#include <stdio.h>
#include <curand.h>
#include <cuda.h>
#include <float.h>
#include "utils.h"
#include "device_utils.h"
#include "kernels.h"
#include "main.h"

#define PRINT_SUMMARY 1
#define DEBUG 1
#define DEBUG_BLOCK 0
#define PROVIDE_SOL 0

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

#if DEBUG
	int *pop = (int*) malloc(N_NODES*POPULATION_SIZE*sizeof(int));
	cudaMemcpy( pop, d_population, N_NODES*POPULATION_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

	for(int t=0; t<POPULATION_SIZE; ++t){
		for(int s =0; s<N_NODES; ++s){
			printf("%d ", pop[t*N_NODES + s]);
		}
		printf("\n");
	}
	int *off = (int*) malloc(N_NODES*POPULATION_SIZE*OFFSPRING_FACTOR*sizeof(int));
#endif
	
	//kernel parameters
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);

	dim3 threadsP(32,prop.maxThreadsDim[0]/32,1);
	dim3 blocksP(ceil(POPULATION_SIZE/prop.maxThreadsDim[0]),1,1); 
	if(POPULATION_SIZE< prop.maxThreadsDim[0]){ 
		threadsP.y = (POPULATION_SIZE < 32 ? 1 : POPULATION_SIZE/32);
		blocksP.x = 1;
	}
	if(POPULATION_SIZE<32 ){
		threadsP.x = POPULATION_SIZE;
	}


	dim3 threadsS(32,prop.maxThreadsDim[0]/32,1);
	dim3 blocksS(ceil((POPULATION_SIZE*OFFSPRING_FACTOR)/prop.maxThreadsDim[0]),1,1); 
	if(POPULATION_SIZE*OFFSPRING_FACTOR< prop.maxThreadsDim[0]){ 
		threadsS.y = (POPULATION_SIZE*OFFSPRING_FACTOR < 32 ? 1 : POPULATION_SIZE*OFFSPRING_FACTOR/32);
		blocksS.x = 1;
	}
	if(POPULATION_SIZE*OFFSPRING_FACTOR<32 ){
		threadsS.x = POPULATION_SIZE*OFFSPRING_FACTOR;
	}
#if DEBUG_BLOCK
	blocksP.x *=2;
	threadsP.y /=2;
	if(threadsP.y ==0) threadsP.y = 1;
	blocksS.x *=2;
	threadsS.y /=2;
	if(threadsS.y ==0) threadsS.y = 1;
#endif
	printf("operation on population will be launched on %d blocks with dim (%d, %d)\n", blocksP.x, threadsP.x,threadsP.y);
	printf("operation on offspring will be launched on %d blocks with dim (%d, %d)\n", blocksS.x, threadsS.x,threadsS.y);

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
		naive_generation<<<blocksP, threadsP>>>(d_population, 
							POPULATION_SIZE, 
							N_NODES, 
							d_offspring, 
							OFFSPRING_FACTOR, 
							d_genetic_rands);
	
		
		printf("it %d: applying selection\n", t);
		naive_selection(d_offspring,
				d_population,
				N_NODES,
				POPULATION_SIZE,
				OFFSPRING_FACTOR,
				d_vec_graph,
				d_fitness,
				d_auxiliary,
				threadsS,
				blocksS,
				threadsP,
				blocksP);

#if DEBUG
		cudaMemcpy( pop, d_population, N_NODES*POPULATION_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

		for(int k=0; k<POPULATION_SIZE; ++k){
			for(int s =0; s<N_NODES; ++s){
				printf("%d ", pop[k*N_NODES + s]);
			}
			printf("fitness: %.2f\n", evaluate_individual_host(vec_graph,N_NODES,pop+k*N_NODES));
		}
#endif	

		//record best solution
		cudaMemcpy( current_best, d_population, N_NODES*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy( &current_fitness, d_fitness, sizeof(float), cudaMemcpyDeviceToHost);
		
		printf("it %d: currently found fitness is %.2f\n", t, current_fitness);

		fitnesses[t] = current_fitness;
		if(current_fitness<best_fitness){
			printf("it %d: improvement found!\n", t);
			best_fitness = current_fitness;
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

#if PROVIDE_SOL
	int sol[] = {0,2,6,4,3,5,1,7};
	printf("best solution computed has lengths %.2f\n",evaluate_individual_host(vec_graph,N_NODES,sol));
	printf("best solution host has lengths %.2f\n",evaluate_individual_host(vec_graph,N_NODES,global_best));
#endif
	free(vec_graph);
	free(global_best);
	free(current_best);
	cudaFree(d_population);
	cudaFree(d_offspring);
	cudaFree(d_shuffle_rands);
	cudaFree(d_genetic_rands);
	cudaFree(d_auxiliary);
#if DEBUG
	free(pop);
	free(off);
#endif

}















