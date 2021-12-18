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
