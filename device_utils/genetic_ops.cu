#include <curand.h>
#include <stdbool.h>


#define DEBUG_PRINT 0

/*
	some infos on the used data:

		- 	the vector storing the main individual from which offspring is generated (here called
			either vec or parent1) is a portion of vector of global or shared memory that represents
			the output vector of the kernel. Before these calls, it is initialized to the corresponding values 
			of the actual parent stored in the input vector of the kernel, hence why it can be modified in place
			without compromising the input population.

		-	random_nums represents a portion (of size 3) of a larger array made of random numbers. 
			There are 3*OFFSPRING_FACTOR random numbers per warp, so that at each call the whole warp is using the
			same random numbers (instruction flow depends on them), and so that the same warp uses a different set
			of random numbers between different calls (that amounts to OFFSPRING_SIZE times, otherwise increasing
			OFFSPRING_SIZE would just generate the same child but replicated).
*/

__device__ void swap_mutation(int *vec, int n_dim, unsigned int* random_nums){
	/*
		mutation genetic operator: two random indexes are chosen, 
		and their content are swapped
		
		random_nums is a small array containing random numbers,
		which are needed to introduce some randomicity
	*/
	unsigned int index1 = random_nums[0]%n_dim;
	unsigned int index2 = random_nums[1]%n_dim;

#if DEBUG_PRINT
	printf("randomicity through %d, %d, i1: %d, i2: %d\n", random_nums[0], random_nums[1], index1, index2);
#endif
	
	int tmp = vec[index1];
	vec[index1] = vec[index2];
	vec[index2] = tmp;
}

__device__ void inversion_mutation(int *vec, int n_dim, unsigned int* random_nums){
	/*
		mutation genetic operator: two random indexes are chosen, 
		all the content between them is inverted
		
	*/
	unsigned int index1 = random_nums[0]%n_dim;
	unsigned int index2 = random_nums[1]%n_dim;
	
	int incr = (index1>index2 ? -1 : 1);
	
	bool go = index1<index2; 
	
#if DEBUG_PRINT
	printf("randomicity through %d, %d, i1: %d, i2: %d\n", random_nums[0], random_nums[1], index1, index2);
#endif

	while (go==(index1<index2)){
		int tmp = vec[index1];
		vec[index1] = vec[index2];
		vec[index2] = tmp;
		index1+=incr;
		index2-=incr;
	}
}

__device__ void cycle_crossover(	int *parent1, 
					int *population, 
					int population_dim, 
					int n_dim,
					unsigned int *random_nums, 
					int p1_i
){
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
	unsigned int p2_i = random_nums[0] % (population_dim*n_dim);
	unsigned int index1 = random_nums[1]%n_dim;
	unsigned int index2 = random_nums[2]%n_dim;


	//make sure that the whole warp is not using the same parent2 to generate offspring
	p2_i = (p2_i + ((p1_i*100)/17) + 1)%population_dim;


	
	//make sure index1 <= index2
	index1 = (index1>index2 ? index2 : index1);
	index2 = (index1>index2 ? index1 : index2);

#if DEBUG_PRINT
	printf("function init: p2_i %d, p1_i %d, i1 %d, i2 %d\n", p2_i, p1_i, index1, index2);
#endif
	
	int assign_i = 0 + (index2+1*(index2+1<n_dim))*(index1==0);
	bool surpassed = 0 + 1*(index1==0);
	
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
				(assign_i<index1 ||(assign_i>index2 && assign_i+1<n_dim)))//regular +1 (should happen often)
				*(!present);						//increment is 0 if we reached the end of p1
											//or we didn't assign the p2 value
		surpassed += assign_i > index1;
#if DEBUG_PRINT
		printf("it: %d, Surpassed: %d, assign_i: %d, present: %d, val: %d\n", t, surpassed, assign_i, present, val);
#endif
		
	}
	
}








