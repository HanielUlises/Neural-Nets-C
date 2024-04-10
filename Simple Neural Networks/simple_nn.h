#ifndef SIMPLE_NN_H
#define SIMPLE_NN_H

#include <stdlib.h>

double single_in_single_out (double input, double weight);
double multiple_in_single_out(double* input, double* weight, int length);
void single_in_multiple_out (double scalar, double* w_vect, double* out_vect, int length);

double weighted_sum (double* input, double* weight, int length);
void element_wise_multiply (double input_scalar, double* weight_vector, double* output_vector, int length);


#endif
