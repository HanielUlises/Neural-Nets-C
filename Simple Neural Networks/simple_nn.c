#include "simple_nn.h"


double single_in_single_out (double input, double weight){
    return(input * weight);
}

double weighted_sum (double* input, double* weight, int length){
    double output;

    for(size_t i = 0; i < length; i++){
        output += input[i] * weight[i];
    }

    return output;
}

void element_wise_multiply (double input_scalar, double* weight_vector, double* output_vector, int length){
    for(size_t i = 0; i < length; i++){
        output_vector[i] = input_scalar * weight_vector[i];
    }
}

void single_in_multiple_out (double scalar, double* w_vect, double* out_vect, int length){
    element_wise_multiply(scalar, w_vect, out_vect, length);
}
