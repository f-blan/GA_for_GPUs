#define DEBUG_PRINT 0
#define SWAP_FACTOR 1
#define INVERT_FACTOR 2
#define RECOMBINATION_FACTOR 6

__global__ void naive_generation(int* population,  
				int *offspring,
				unsigned int *random_nums)
{
	
	unsigned int tid = blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x+ threadIdx.x;
#if DEBUG_PRINT
	printf("tid: %d working from position %d, rand: %d\n", tid, tid*N_NODES*OFFSPRING_FACTOR,(int) random_nums[0] );
#endif
	
	//copy the parent in all children positions in the output vector
	unsigned int t;
	for(t = 0; t<N_NODES*OFFSPRING_FACTOR; ++t){
		offspring[tid*N_NODES*OFFSPRING_FACTOR + t] = population[tid*N_NODES +t%N_NODES];
	}


	if(( blockIdx.x*blockDim.y + (int)threadIdx.y)%RECOMBINATION_FACTOR <= SWAP_FACTOR){
#if DEBUG_PRINT
		printf("thread from warp: %d performing swap\n",threadIdx.y);
#endif		
		for(t=0; t< OFFSPRING_FACTOR; ++t){
			
			swap_mutation(	offspring+(tid*N_NODES*OFFSPRING_FACTOR +N_NODES*t), 
					N_NODES, 
					random_nums + ((tid/32)*3 + 3*t));
		}
	}else if(( blockIdx.x*blockDim.y + (int)threadIdx.y)%RECOMBINATION_FACTOR <= INVERT_FACTOR){
#if DEBUG_PRINT
		printf("thread from warp: %d performing inversion\n", threadIdx.y);
#endif	
		for(t=0; t< OFFSPRING_FACTOR; ++t){
			
			inversion_mutation(	offspring+(tid*N_NODES*OFFSPRING_FACTOR +N_NODES*t), 
						N_NODES, 
						random_nums + ((tid/32)*3 + 3*t));

		}
	}else{
#if DEBUG_PRINT
		printf("thread from warp: %d performing cycle crossover\n", threadIdx.y);
#endif
		for(t=0; t< OFFSPRING_FACTOR; ++t){
			
			cycle_crossover(	offspring+(tid*N_NODES*OFFSPRING_FACTOR +N_NODES*t), 
						population,
						POPULATION_SIZE,
						N_NODES, 
						random_nums + ((tid/32)*3 + 3*t),
						tid);
		}
	}
}




__global__ void shared_generation(	int* population, 
					int *offspring,
					unsigned int *random_nums)
{
#if COMPILE_SHARED
	//shared vector containing both the population and the offspring
	//printf("hello\n");
	__shared__ int s_off[N_NODES*THREADS_PER_BLOCK*OFFSPRING_FACTOR];
	unsigned int tid = blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x+ threadIdx.x;
	unsigned int tid_b = threadIdx.y*blockDim.x+ threadIdx.x;
	
	//copy the parent in all children and its respective position in the shared vector from global memory
	unsigned int t;
	int val;
	int s;
	for(t =0; t<N_NODES*OFFSPRING_FACTOR; ++t){
		s_off[tid_b*N_NODES*OFFSPRING_FACTOR + t] = population[tid*N_NODES + t];
	}
	/*for(t = 0; t<N_NODES; ++t){
		val = s_off[tid_b*N_NODES*OFFSPRING_FACTOR + t];

		for(s=1; s<OFFSPRING_FACTOR;++s){
			s_off[tid_b*N_NODES*OFFSPRING_FACTOR + s*N_NODES + t] = val;
		
		}
	}*/
	__syncthreads();

	//perform genetic ops
	if(( blockIdx.x*blockDim.y + (int)threadIdx.y)%RECOMBINATION_FACTOR <= SWAP_FACTOR){
#if DEBUG_PRINT
		printf("thread from warp: %d performing swap\n",threadIdx.y);
#endif		
		for(t=0; t< OFFSPRING_FACTOR; ++t){
			
			swap_mutation(	s_off+(tid_b*N_NODES*OFFSPRING_FACTOR +N_NODES*t), 
					N_NODES, 
					random_nums + ((tid/32)*3 + 3*t));
		}
	}else if(( blockIdx.x*blockDim.y + (int)threadIdx.y)%RECOMBINATION_FACTOR <= INVERT_FACTOR){
#if DEBUG_PRINT
		printf("thread from warp: %d performing inversion\n", threadIdx.y);
#endif	
		for(t=0; t< OFFSPRING_FACTOR; ++t){
			
			inversion_mutation(	s_off+(tid_b*N_NODES*OFFSPRING_FACTOR +N_NODES*t), 
						N_NODES, 
						random_nums + ((tid/32)*3 + 3*t));

		}
	}else{
#if DEBUG_PRINT
		printf("thread from warp: %d performing cycle crossover\n", threadIdx.y);
#endif
		for(t=0; t< OFFSPRING_FACTOR; ++t){

#if USE_ISLAND_GENERATION			
			cycle_crossover(	s_off+(tid_b*N_NODES*OFFSPRING_FACTOR +N_NODES*t), 
						population + blockDim.y*blockDim.x*blockIdx.x*N_NODES,
						blockDim.x*blockDim.y,
						N_NODES, 
						random_nums + ((tid/32)*3 + 3*t),
						tid);
#else
			cycle_crossover(	s_off+(tid_b*N_NODES*OFFSPRING_FACTOR +N_NODES*t), 
						population,
						POPULATION_SIZE,
						N_NODES, 
						random_nums + ((tid/32)*3 + 3*t),
						tid);
#endif
		}
	}
	__syncthreads();
	//copy back in global arrays
	for(t=0; t<OFFSPRING_FACTOR; ++t){
		for(s=0; s<N_NODES; ++s){
			offspring[tid*N_NODES*OFFSPRING_FACTOR + t*N_NODES + s] = s_off[tid_b*N_NODES + t*N_NODES + s];
		}
	}


#endif
	
}



