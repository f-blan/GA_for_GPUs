#include "device_utils.cu"
#include "genetic_ops.cu"

void say_hi_cu(void);
void swap_mutation(int * vec, int n_dim, int *random_nums);
void inversion_mutation(int *vec, int n_dim, int* random_nums);
void cycle_crossover(int *parent1, int *population, int population_dim, int n_dim, int *random_nums, int p1_i);
