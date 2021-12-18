#include <stdio.h>
#define PRINT_DEBUG 0

__device__ float evaluate_individual(float *graph, int n_dim, int *individual){
	float cost =0;	

	for(int t=0; t<n_dim-1; ++t){
#if PRINT_DEBUG 
		printf("%d -> %d\n",individual[t], individual[t+1]);
		printf("evaluation, cost1 is %.2f for %d -> %d\n", graph[individual[t]*n_dim + individual[t+1]], individual[t], individual[t+1]);
#endif
		cost += graph[individual[t]*n_dim + individual[t+1]];
	}
	cost += graph[individual[0]*n_dim + individual[n_dim-1]];
	return cost;
}
