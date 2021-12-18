#define DEBUG_PRINT 1

__global__ void naive_selection(	int *offspring, 
					int *next_generation, 
					int n_dim, 
					int population_dim,
					int offspring_factor,
					float *graph,
					float *fitness,
					int *auxiliary)
{
	unsigned int tid = blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x+ threadIdx.x;
	
	//evaluate each child
	fitness[tid] = evaluate_individual(graph, n_dim, offspring + tid);
	auxiliary[tid] =tid;
#if DEBUG_PRINT	
	printf("%d got %.2f\n", tid, fitness[tid]);
#endif
	__syncthreads();
	unsigned int tid_idx,tmp;
	float d_0, d_1;
	unsigned int offset = 0;
	unsigned int tid_max = (offspring_factor*population_dim -1);
	int i;
	//sort the children
	for(i = 0; i< population_dim*offspring_factor; ++i){
		tid_idx = (tid*2) + offset;
		if(tid_idx < tid_max){
			d_0 = fitness[tid_idx];
			d_1 = fitness[tid_idx + 1];
			if(d_0 > d_1){
				fitness[tid_idx] = d_1;
				fitness[tid_idx+1] = d_0;
				
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
#if DEBUG_PRINT
	printf("tid: %d, place: %d, val: %.1f\n", tid, auxiliary[tid], fitness[tid]);
#endif
	
	//store back into pop only the top N
	if(tid<population_dim){
		for(i=0; i<n_dim; ++i){
			next_generation[tid*n_dim + i]= offspring[n_dim*auxiliary[tid] + i];
		}
	}
}


