#include <stdio.h>
#include <curand.h>
#include <cuda.h>
#include "utils.h"
#include "device_utils.h"
#include "kernels.h"
#include "main.h"



void test_utils();
void test_gen_ops();
void test_curand();

int main(void){
	//test_utils();
	//test_gen_ops();
	test_curand();

	return 0;
}

void test_utils(){
	float **g = graph_generate(N_NODES);
	print_graph(g, N_NODES);
	
	float **m = graph_to_mat(g, N_NODES);
	//print_mat(m, N_NODES);
}

void test_gen_ops(){
	int vec[] = {0,1,2,3,4};
	swap_mutation(vec);

}

void test_curand(){
	int size = 32;

	curandGenerator_t gen;

	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

	int *d_r; 
	cudaMalloc((void **) &d_r, size*sizeof(int));

	int *r =(int *) malloc(size*sizeof(int));

	curandSetPseudoRandomGeneratorSeed(gen, 1234);

	curandGenerate(gen, (unsigned int *) d_r, size*sizeof(int));
	
	cudaMemcpy(r, d_r, size*sizeof(int), cudaMemcpyDeviceToHost);
	
	for(int i =0; i<size; ++i){
		printf("%d ", r[i]);
	}
	printf("\n");
	
	curandDestroyGenerator(gen);
	cudaFree(d_r);
	free(r);
	
}











