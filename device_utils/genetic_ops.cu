#include <curand.h>
#include <stdbool.h>

#define DEBUG_PRINT 1

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

void cycle_crossover(int *parent1, int *population, int population_dim, int n_dim, int *random_nums, int p1_i){
	/*
		recombination genetic operator: two random indexes are chosen,
		everything between them is kept, the rest of the array is reconstructed
		from second parent
		
		es.
		0 1 |2 3 4| 5
				-> 5 1 2 3 4 0 
		5 4 3 2 1 0

		this function might very easily lead to warp divergence (instruction flow is
		dependent on the data itself). As a design choice, instructions that should
		lead to warp divergence for the current thread instead perform a no-op
	*/
	
	//find second parent and random indexes
	int p2_i = random_nums[0] % (population_dim*n_dim);
	int index1 = random_nums[1]%n_dim;
	int index2 = random_nums[2]%n_dim;

#if DEBUG_PRINT
	printf("function init: p2_i %d, i1 %d, i2 %d\n", p2_i, index1, index2);
#endif

	//make sure that the whole warp is not using the same parent2 to generate offspring
	p2_i = (p2_i + p1_i)%population_dim;

	//make sure that the second parent is not the exactly the first parent
	if(p2_i == p1_i){
		p2_i = (p1_i == 0 ? p1_i+1 : p1_i-1);
	}
	
	//make sure index1 <= index2
	index1 = (index1>index2 ? index2 : index1);
	index2 = (index1>index2 ? index1 : index2);
	
	int assign_i = 0;
	bool surpassed = 0;
	
	for(int t = 0; t<n_dim; ++t){
		//iterate over p2
		int val = population[p2_i*n_dim +t];
		bool present = 0;
		
		//check if p2[t] is eligible to be assigned
		for(int k = index1; k<=index2 ; ++k){
			present += val == parent1[k];
		}
		//assign to parent1[assign_i] if this value is not in the randomly chosen interval
		//otherwise assign same value as before		
		parent1[assign_i] = parent1[assign_i]*present + val*((int) !present);
		
		//update assign_i while taking care if assign_i has surpassed index1
		assign_i += 	((assign_i+1 >= index1)*(index2-index1+1*(index2+1 < n_dim))
								*(!surpassed) + 	//surpassing increment (happens once)
				(assign_i<index1 ||(assign_i>index2 && assign_i<n_dim)))//regular +1 (should happen often)
				*(!present);						//increment is 0 if we reached the end of p1
											//or we didn't assign the p2 value
		surpassed += assign_i > index1;
#if DEBUG_PRINT
		printf("it: %d, Surpassed: %d, assign_i: %d, present: %d, val: %d\n", t, surpassed, assign_i, present, val);
#endif
		
	}
	
}








