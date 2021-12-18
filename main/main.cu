#include <stdio.h>
#include "utils.h"
#include "device_utils.h"
#include "kernels.h"
#include "main.h"

int main(void){
	float **g = graph_generate(N_NODES);
	print_graph(g, N_NODES);
	//view_graph(g, N_NODES);
	float **m = graph_to_mat(g, N_NODES);
	print_mat(m, N_NODES);
	
	return 0;
}
