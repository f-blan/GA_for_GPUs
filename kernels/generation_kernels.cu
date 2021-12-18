#define DEBUG_PRINT 1
#define SWAP_FACTOR -1
#define INVERT_FACTOR -1
#define RECOMBINATION_FACTOR 6

__global__ void naive_generation(int* population, 
				int population_dim, 
				int n_dim, int *offspring,
				int offspring_factor,
				unsigned int *random_nums)
{
	
	unsigned int tid = blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*32+ threadIdx.x;
#if DEBUG_PRINT
	printf("tid: %d working from position %d, rand: %d\n", tid, tid*n_dim*offspring_factor,(int) random_nums[0] );
#endif
	
	//copy the parent in all children positions in the output vector
	unsigned int t;
	for(t = 0; t<n_dim*offspring_factor; ++t){
		offspring[tid*n_dim*offspring_factor + t] = population[tid*n_dim +t%n_dim];
	}


	if((int)threadIdx.y%RECOMBINATION_FACTOR <= SWAP_FACTOR){
#if DEBUG_PRINT
		printf("thread from warp: %d performing swap\n",threadIdx.y);
#endif		
		for(t=0; t< offspring_factor; ++t){
			
			swap_mutation(	offspring+(tid*n_dim*offspring_factor +n_dim*t), 
					n_dim, 
					random_nums + ((tid/32)*3 + 3*t));
		}
	}else if((int)threadIdx.y%RECOMBINATION_FACTOR <= INVERT_FACTOR){
#if DEBUG_PRINT
		printf("thread from warp: %d performing inversion\n", threadIdx.y);
#endif	
		for(t=0; t< offspring_factor; ++t){
			
			inversion_mutation(	offspring+(tid*n_dim*offspring_factor +n_dim*t), 
						n_dim, 
						random_nums + ((tid/32)*3 + 3*t));

		}
	}else{
#if DEBUG_PRINT
		printf("thread from warp: %d performing cycle crossover\n", threadIdx.y);
#endif
		for(t=0; t< offspring_factor; ++t){
			
			cycle_crossover(	offspring+(tid*n_dim*offspring_factor +n_dim*t), 
						population,
						population_dim,
						n_dim, 
						random_nums + ((tid/32)*3 + 3*t),
						tid*n_dim*offspring_factor +n_dim*t);
		}
	}
}
