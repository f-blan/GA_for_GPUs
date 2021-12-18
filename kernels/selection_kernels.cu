#define DEBUG_PRINT 0

#pragma once
__constant__ float const_graph[N_NODES*N_NODES];

__global__ void evaluate_kernel(	int *offspring,  
					int n_dim, 
					const float *graph,
					float *fitness,
					int *auxiliary)
{
	unsigned int tid = blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x+ threadIdx.x;
	
	//evaluate each child
	fitness[tid] = evaluate_individual(graph, n_dim, offspring + (tid*n_dim));
	auxiliary[tid] =tid;
	float val = const_graph[0];
#if DEBUG_PRINT	
	printf("%d got %.2f\n", tid, fitness[tid]);
#endif
}

__global__ void constant_evaluate_kernel(	int *offspring,  
						int n_dim,
						float *fitness,
						int *auxiliary)
{
	unsigned int tid = blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x+ threadIdx.x;
	
	//evaluate each child
	fitness[tid] = evaluate_individual(const_graph, n_dim, offspring + (tid*n_dim));
	auxiliary[tid] =tid;
	float val = const_graph[0];
#if DEBUG_PRINT	
	printf("%d got %.2f\n", tid, fitness[tid]);
#endif
}

void naive_selection(	int *offspring, 
			int *next_generation, 
			int n_dim, 
			int population_dim,
			int offspring_factor,
			float *graph,
			float *fitness,
			int *auxiliary,
			dim3 threadsS,
			dim3 blocksS,
			dim3 threadsP,
			dim3 blocksP)
{
	evaluate_kernel<<<blocksS, threadsS>>>(offspring, n_dim, graph, fitness, auxiliary);
	thrust::sort_by_key(thrust::device, fitness, fitness+population_dim*offspring_factor, auxiliary);
	swap_with_positions<<<blocksP, threadsP>>>(  offspring, next_generation, auxiliary, n_dim, population_dim);


}

void const_selection(	int *offspring, 
			int *next_generation, 
			int n_dim, 
			int population_dim,
			int offspring_factor,
			float *fitness,
			int *auxiliary,
			dim3 threadsS,
			dim3 blocksS,
			dim3 threadsP,
			dim3 blocksP)
{
	constant_evaluate_kernel<<<blocksS, threadsS>>>(offspring, n_dim, fitness, auxiliary);
	thrust::sort_by_key(thrust::device, fitness, fitness+population_dim*offspring_factor, auxiliary);
	swap_with_positions<<<blocksP, threadsP>>>(  offspring, next_generation, auxiliary, n_dim, population_dim);


}

__global__ void island_selection(	int *offspring,
					int *population,  
					int n_dim,
					float *fitness
					)
{
#if COMPILE_SHARED
	// an auxiliary vector + the offspring
	__shared__ int s[(THREADS_PER_BLOCK)+ N_NODES*(THREADS_PER_BLOCK)*OFFSPRING_FACTOR];
	int * s_off = s+(POPULATION_SIZE/THREADS_PER_BLOCK);
	//the fitness	
	__shared__ float s_fit[(THREADS_PER_BLOCK)];

	unsigned int tid = blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x+ threadIdx.x;
	unsigned int tid_b = threadIdx.y*blockDim.x+ threadIdx.x;

	//copy into shared memory + evaluation
	int t;
	for(t=0; t<N_NODES; ++t){
		s_off[tid_b*N_NODES +t] = offspring[tid*N_NODES +t]; 
	}
	s[tid_b]= tid_b;
	s_fit[tid_b] = evaluate_individual(const_graph, N_NODES, s_off + (tid_b*N_NODES));

	__syncthreads();
	
	//sort the children within the island (block)
	unsigned int tid_idx;
	float f_0, f_1;
	unsigned int offset = 0;
	unsigned int tid_max = (THREADS_PER_BLOCK -1);
	int tmp;

	for(int i = 0; i< THREADS_PER_BLOCK; ++i){
		tid_idx = (tid_b*2) + offset;
		if(tid_idx < tid_max){
			f_0 = s_fit[tid_idx];
			f_1 = s_fit[tid_idx + 1];
			if(f_0 > f_1){
				s_fit[tid_idx] = f_1;
				s_fit[tid_idx+1] = f_0;

				tmp = s[tid_idx];
				s[tid_idx] = s[tid_idx+1];
				s[tid_idx+1] = tmp;
				
			}
		}
		if(offset == 0){
			offset = 1;
		}else{
			offset = 0;
		}
		__syncthreads();
	}
	//only the best within the block get copied back into next generation
	if(tid_b < THREADS_PER_BLOCK/OFFSPRING_FACTOR){
		tid_idx = s[tid_b];
		for(t=0; t<N_NODES; ++t){
			population[blockIdx.x*(THREADS_PER_BLOCK/OFFSPRING_FACTOR) + tid_b*N_NODES +t] = s_off[tid_idx*N_NODES + t];
		}
		fitness[blockIdx.x*(THREADS_PER_BLOCK/OFFSPRING_FACTOR) + tid_b] = s_fit[tid_b];
		
	}

#endif

}


















