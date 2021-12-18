#define DEBUG_PRINT 0

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

	
	for(t=0; t< offspring_factor; ++t){
#if DEBUG_PRINT
	printf("tid: %d swapping individual at %d, random pos: %d\n, pointer val: %d, start val: %d\n", 
		tid, tid*n_dim*offspring_factor +n_dim*t, (tid/32)*3 + 3*t,
		(offspring+(tid*n_dim*offspring_factor +n_dim*t)), ((int)offspring + 8));
	int * start_off = offspring+(tid*n_dim*offspring_factor +n_dim*t);
	unsigned int * start_rand =  random_nums + (tid/32)*3 + 3*t;
	printf("start: %d, pos: %d, first val: %d\n",(int) offspring,(int) start_off, start_off[0]);
	printf("rstart: %d, rpos: %d, rfirst val: %u\n",(int) random_nums,(int) start_rand, start_rand[0]);
#endif
		swap_mutation(offspring+(tid*n_dim*offspring_factor +n_dim*t), 
		n_dim, 
		random_nums + ((tid/32)*3 + 3*t));
	}
	
}
