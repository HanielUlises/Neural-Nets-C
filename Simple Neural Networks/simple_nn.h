#ifndef SIMPLE_NN_H
#define SIMPLE_NN_H

#include <stdlib.h>

#define NUM_FEATURES 3
#define OUTPUTS 3


double single_in_single_out(double input, double weight);
double multiple_in_single_out(double *input, double *weight, int length);
void single_in_multiple_out(double scalar, double *w_vect, double *ut_vect, int length);
void multiple_in_multiple_out(double *input_vector, int INPUT_LEN, double *output_vector, int OUTPUT_LEN, double weight_matrix[][OUTPUTS]);

double weighted_sum(double *input, double *weight, int length);
void element_wise_multiply(double input_scalar, double *weight_vector, double *output_vector, int length);
void matrix_vector_multiply(double *input_vector, int INPUT_LEN, double *output_vector, int OUTPUT_LEN, double weight_matrix[][OUTPUTS]);

#endif
