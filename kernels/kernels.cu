#include <stdio.h>

#define SWAP_ITERATIONS 2
#define DEBUG_PRINT 0

//simple function to initialize the population pseudo randomly
__global__ void init_pop(int *pop, int pop_dim, int n_dim, unsigned int *random_nums, int r_dim){

	int tid = blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*32+ threadIdx.x;

	//pseudo random initialization based on tid + random_nums
	for(int t =0; t<n_dim; ++t)
		pop[tid*n_dim + t] = t;

#if DEBUG_PRINT
	printf("thread %d taking care of pos %d\nRandom used: %d - %d - %d - %d\n"
	, tid, tid*n_dim, random_nums[tid]);
#endif	
	
	unsigned int random_i = tid;
	for(int t = 0; t< SWAP_ITERATIONS; ++t){
		int random = random_nums[random_i];
		for(int s = 0; s<n_dim-1; ++s){
			if(random & 1){
				//swap				
				int tmp =  pop[tid*n_dim +s +1];
				pop[tid*n_dim +s +1] = pop[tid*n_dim + s];
				pop[tid*n_dim +s] = tmp;
			}
			random << 1;
		}
		if(random_nums[random_i] & 1){
			//swap				
			int tmp =  pop[tid*n_dim +n_dim-1];
			pop[tid*n_dim +n_dim -1] = pop[tid*n_dim+ 0];
			pop[tid*n_dim + 0] = tmp;
		}
		//random_i = (random_i + 1)%r_dim; 
	}
#if DEBUG_PRINT
	printf("thread %d finished the job, %d\n", tid, pop[tid*n_dim + 5]);
#endif	
}


//simple function to initialize the population pseudo randomly

//this performs a bubble sort based on the random numbers generated, and produces the related permutation
//on the individual
__global__ void init_pop_s(int *pop, int pop_dim, int n_dim, unsigned int *random_nums, int r_dim){

	int tid = blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*32+ threadIdx.x;

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




