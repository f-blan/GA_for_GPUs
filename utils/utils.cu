#include <stdio.h>



#define SPACE_WIDTH 100
#define SPACE_HEIGHT 100
#define DEBUG_PRINT 1

void say_hi(void){
	printf("HELLOOOOO\n");
}

float **graph_generate(int n_nodes){
	/*
		generate a random graph of n_nodes. The position of each node is pseudorandomic
		for the sake of reproducibility of the experiments 
	*/	
	float **g = (float **) malloc(n_nodes*sizeof(float *));
	for(int t = 0; t<n_nodes; ++t){
		g[t] = (float *) malloc(2*sizeof(float));
		g[t][0] = (float)rand()/(float)(RAND_MAX/SPACE_HEIGHT);
		g[t][1] = (float)rand()/(float)(RAND_MAX/SPACE_WIDTH);
	}
	return g;
}

void print_graph(float **g, int n_nodes){
	//just prints node coordinates	
	printf("printing nodes:\n");
	for(int t =0; t<n_nodes; ++t){
		printf("%d) ", t);
		printf("%.1f - %.1f\n", g[t][1],g[t][0]);
	}
	printf("-----------\n");
}

void view_graph(float **g, int n_nodes){
	//simple function to approximately visualize the graph
	for(int t=0; t< SPACE_HEIGHT; ++t){
		for(int s=0; s<SPACE_WIDTH; ++s){
			int n_index = -1;
			for(int r=0; r<n_nodes; ++r){
				int x = floor((double) g[r][1]);
				int y = floor((double) g[r][0]);
				if(x==s && y==t){
					n_index = r;
					break;				
				}
			}
			if(n_index!=-1){
				printf("%d ", n_index);
			}else{
				printf("  ");
			}
		}
		printf("\n");
	}
}

float calc_dist(float *n1, float *n2){
	return sqrt((n1[0]-n2[0])*(n1[0]-n2[0]) + (n1[1]-n2[1])*(n1[1]-n2[1]));
}

float **graph_to_mat(float **g, int n_nodes){

	/*
		simple function to turn a graph into an
		adjacency matrix
	*/
	float **m = (float **) malloc(n_nodes*sizeof(float *));
	
	for(int t=0; t<n_nodes; ++t){
		m[t] =(float *) malloc(n_nodes*sizeof(float));
		for(int s =0; s<n_nodes; ++s){
			m[t][s] = calc_dist(g[t], g[s]);
		}
	}
	return m;
	
}
float *mat_to_vec(float **m, int n_nodes){
	float *v = (float *) malloc(n_nodes*n_nodes*sizeof(float));
	
	for(int t =0; t<n_nodes; ++t){
		for(int s =0; s<n_nodes; ++s){
			v[t*n_nodes + s] = m[t][s];
		}
	}
	return v;
}

void print_mat(float **m, int dim){
	for(int t =0; t<dim; ++t){
		for(int s =0; s<dim;++s){
			printf("%.1f ", m[t][s]);
		}
		printf("\n");
	}
}

float evaluate_individual_host(float *graph, int n_dim, int *individual){
	float cost =0;	

	for(int t=0; t<n_dim-1; ++t){
#if DEBUG_PRINT  
		printf("evaluation, cost1 is %.2f for %d -> %d\n", graph[individual[t]*n_dim + individual[t+1]], individual[t], individual[t+1]);
#endif
		cost += graph[individual[t]*n_dim + individual[t+1]];
	}

#if DEBUG_PRINT  
		printf("evaluation, cost1 is %.2f for %d -> %d\n", graph[individual[n_dim-1]*n_dim + individual[0]], individual[n_dim-1], individual[0]);
#endif
	cost += graph[individual[0]*n_dim + individual[n_dim-1]];
	return cost;
}





















