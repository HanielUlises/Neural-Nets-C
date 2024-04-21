#ifndef SIMPLE_NN_H
#define SIMPLE_NN_H

#include <stdlib.h>

#define NUM_FEATURES 3
#define OUTPUTS 3

#define HIDDEN_SIZE 3
#define INPUT_SIZE 3
#define OUTPUT_SIZE 3

// Neural networks
double single_in_single_out(double input, double weight);
double multiple_in_single_out(double *input, double *weight, int length);
void single_in_multiple_out(double scalar, double *w_vect, double *ut_vect, int length);
void multiple_in_multiple_out(double *input_vector, int INPUT_LEN, double *output_vector, int OUTPUT_LEN, double weight_matrix[][OUTPUTS]);


void hidden_layer_nn(double *input_vector, 
                     double in_to_hid_weights[HIDDEN_SIZE][INPUT_SIZE], 
                     double hid_to_out_weights[OUTPUT_SIZE][HIDDEN_SIZE], 
                     double *output_vector);


// Utils
double weighted_sum(double *input, double *weight, int length);
void element_wise_multiply(double input_scalar, double *weight_vector, double *output_vector, int length);
void matrix_vector_multiplication(double *input_vector, int INPUT_LEN, double *output_vector, int OUTPUT_LEN, double weight_matrix[][OUTPUTS]);

#endif
