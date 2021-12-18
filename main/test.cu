#include <stdio.h>
#include <curand.h>
#include <cuda.h>
#include "utils.h"
#include "device_utils.h"
#include "kernels.h"
#include "main.h"



void test_utils();
void test_mutation_ops();
void test_curand();
void test_cycle_CO();

int main(void){
	//test_utils();
	test_mutation_ops();
	//test_curand();
	test_cycle_CO();

	return 0;
}

void test_cycle_CO(){
	int size = 6; 
	int vec[] = { 	
			0,1,2,3,4,5,
			1,2,3,4,5,0			
		};
	unsigned int rands[] = { 1, 1, 4 };
	cycle_crossover(vec, vec, 2, size, rands, 0);

	for (int t = 0; t<2; ++t){
		for(int s =0; s<size; ++s){
			printf("%d ", vec[t*size +s]);
		}
		printf("\n");
	}

	int vec2[] = { 	
			0,1,2,3,4,5,
			1,2,3,4,5,0			
		};
	unsigned int rands2[] = { 1, 1, 3 };
	cycle_crossover(vec2, vec2, 2, size, rands2, 0);

	for (int t = 0; t<2; ++t){
		for(int s =0; s<size; ++s){
			printf("%d ", vec2[t*size +s]);
		}
		printf("\n");
	}

	int vec3[] = { 	
			0,1,2,3,4,5,
			1,2,3,4,5,0			
		};
	unsigned int rands3[] = { 1, 0, 3 };
	cycle_crossover(vec3, vec3, 2, size, rands3, 0);

	for (int t = 0; t<2; ++t){
		for(int s =0; s<size; ++s){
			printf("%d ", vec3[t*size +s]);
		}
		printf("\n");
	}
}

void test_utils(){
	float **g = graph_generate(N_NODES);
	print_graph(g, N_NODES);
	
	float **m = graph_to_mat(g, N_NODES);
	//print_mat(m, N_NODES);
}

void test_mutation_ops(){
	int vec[] = {0,1,2,3,4};
	unsigned int rands[4];
	printf("rands\n");
	for(int t = 0; t< 4; ++t){
		rands[t] = rand();
		printf("%d ", rands[t]);
	}
	printf("\n");
	
	swap_mutation(vec, 5, rands);
	printf("swap\n");
	for(int t = 0; t< 5; ++t){
		printf("%d ", vec[t]);
		vec[t] = t;
	}
	printf("\n");
	
	unsigned int rands2[] = {4,0,1,2,3}; 
	inversion_mutation(vec, 5, rands2);
	printf("invert\n");
	for(int t = 0; t< 5; ++t){
		printf("%d ", vec[t]);
		vec[t] = t;
	}
	printf("\n");

	

}

void test_curand(){
	int size = 32;

	curandGenerator_t gen;

	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

	unsigned int *d_r; 
	cudaMalloc((void **) &d_r, size*sizeof(int));

	unsigned int *r =(unsigned int *) malloc(size*sizeof(int));

	curandSetPseudoRandomGeneratorSeed(gen, 1234);

	curandGenerate(gen, (unsigned int *) d_r, size*sizeof(int));
	
	cudaMemcpy(r, d_r, size*sizeof(int), cudaMemcpyDeviceToHost);
	
	for(int i =0; i<size; ++i){
		printf("%u ", r[i]%size);
	}
	printf("\n");
	
	curandDestroyGenerator(gen);
	cudaFree(d_r);
	free(r);
	
}











