#include <curand.h>
#include <stdbool.h>

void swap_mutation(int *vec, int n_dim, int* random_nums){
	/*
		mutation genetic operator: two random indexes are chosen, 
		and their content are swapped
		
		random_nums is a small array containing random numbers,
		which are needed to introduce some randomicity
	*/
	int index1 = random_nums[0]%n_dim;
	int index2 = random_nums[1]%n_dim;
	
	int tmp = vec[index1];
	vec[index1] = vec[index2];
	vec[index2] = tmp;
}

void inversion_mutation(int *vec, int n_dim, int* random_nums){
	/*
		mutation genetic operator: two random indexes are chosen, 
		all the content between them is inverted
		
	*/
	int index1 = random_nums[0]%n_dim;
	int index2 = random_nums[1]%n_dim;
	
	int incr = (index1>index2 ? -1 : 1);
	
	bool go = index1<index2; 
	
	while (go==(index1<index2)){
		int tmp = vec[index1];
		vec[index1] = vec[index2];
		vec[index2] = tmp;
		index1+=incr;
		index2-=incr;
	}
}
