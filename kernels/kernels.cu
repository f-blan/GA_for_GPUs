#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#define SWAP_ITERATIONS 2
#define DEBUG_PRINT 0




//simple function to initialize the population pseudo randomly

//this performs a bubble sort based on the random numbers generated, and produces the related permutation
//on the individual
__global__ void init_pop_s(int *pop, int pop_dim, int n_dim, unsigned int *random_nums){

	int tid = blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x+ threadIdx.x;

	//pseudo random initialization based on tid + random_nums
	for(int t =0; t<n_dim; ++t)
		pop[tid*n_dim + t] = t;

	
	unsigned int random_start = tid*n_dim;
	int tmp;
	for(int t=0; t<n_dim-1; ++t){
		for(int s =0; s<n_dim-1-t; ++s){
			if(random_nums[random_start + s] > random_nums[random_start +s +1]){
				tmp = random_nums[random_start +s ];
				random_nums[random_start +s] = random_nums[random_start +s +1];
				random_nums[random_start +s +1] = tmp;
				
				tmp = pop[tid*n_dim +s ];
				pop[tid*n_dim +s] = pop[tid*n_dim +s +1];
				pop[tid*n_dim +s +1] = tmp;
			}
		}
	}
}

//simple shuffle algorithm

__global__ void shuffle(int *population,int*out, int population_dim, int n_dim, int*auxiliary, unsigned int *rands){
	unsigned int tid = blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x+ threadIdx.x;	
	
	unsigned int tid_idx,tmp;
	unsigned int d_0, d_1;
	unsigned int offset = 0;
	unsigned int tid_max = (population_dim -1);
	int i;
	auxiliary[tid] = tid;



	__syncthreads();
	//sort the children
	for(i = 0; i< population_dim; ++i){
		tid_idx = (tid*2) + offset;
		if(tid_idx < tid_max){
			d_0 = rands[tid_idx];
			d_1 = rands[tid_idx + 1];
			if(d_0 < d_1){
				rands[tid_idx] = d_1;
				rands[tid_idx+1] = d_0;
				
				tmp = auxiliary[tid_idx];
				auxiliary[tid_idx] = auxiliary[tid_idx+1];
				auxiliary[tid_idx+1] = tmp;

			}
		}
		if(offset == 0){
			offset = 1;
		}else{
			offset = 0;
		}
		__syncthreads();
	}
	int val;
	for(i=0; i<n_dim; ++i){
		
		out[tid*n_dim + i]=population[n_dim*auxiliary[tid] + i];
	}
}
__global__ void swap_with_positions(int *copy, int *out, int *positions, int n_dim, int population_dim){
	unsigned int tid = blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x+ threadIdx.x;

	int pos = positions[tid];

	for(int t=0; t<n_dim; ++t){
		out[tid*n_dim + t] = copy[pos*n_dim+t];
	}

}

//a way to randomly shuffle the population. Requires some auxiliary vectors
void thrust_shuffle(int *pop,int * copy, int *positions, curandGenerator_t gen, unsigned int * rands, int n_dim, int population_dim){
	
	cudaMemcpy( copy, pop, population_dim*n_dim*sizeof(int), cudaMemcpyDeviceToDevice);
	
	curandGenerate(gen, (unsigned int *) rands, population_dim*sizeof(unsigned int));


	thrust::sort_by_key(thrust::device, rands, rands+population_dim, positions);
	


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
	swap_with_positions<<<blocks, threads>>>(  copy, pop, positions, n_dim, population_dim);
}

__global__ void swap_best(int *population, float * fitness, int best_pos, int *global_best_sol, float *global_best_fit){
	
	int tid = threadIdx.x;
	if(fitness[best_pos] < global_best_fit[0]){
		global_best_sol[tid] = population[best_pos*N_NODES + tid];
		if(tid ==0){
			global_best_fit[0] = fitness[best_pos];
		}
	}


}




