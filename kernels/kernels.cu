#include <stdio.h>

#define SWAP_ITERATIONS 2
#define DEBUG_PRINT 1



//simple function to initialize the population pseudo randomly

//this performs a bubble sort based on the random numbers generated, and produces the related permutation
//on the individual
__global__ void init_pop_s(int *pop, int pop_dim, int n_dim, unsigned int *random_nums){

	int tid = blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x+ threadIdx.x;

	//pseudo random initialization based on tid + random_nums
	for(int t =0; t<n_dim; ++t)
		pop[tid*n_dim + t] = t;

#if DEBUG_PRINT
	printf("thread %d taking care of pos %d\nRandom used: %d \n", tid, tid*n_dim,
	 	random_nums[tid*n_dim]);
#endif	
	
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
#if DEBUG_PRINT
	if(tid == 0 ){
		printf("%d ", tid);
		for(int t =0; t<n_dim; ++t){
			printf("%d ",random_nums[random_start +t] );
		}
		printf("\n");
	}
#endif	
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

#if DEBUG_PRINT
	printf("tid: %d, rand: %10u\n", tid, rands[tid] );
#endif

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




