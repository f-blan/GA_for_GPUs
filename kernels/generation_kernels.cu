

__global__ void naive_generation(int* population, 
				int population_dim, 
				int n_dim, int *offspring, 
				unsigned int *random_nums)
{
	int tid = blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*32+ threadIdx.x;
	printf("hello from %d", tid);
}
