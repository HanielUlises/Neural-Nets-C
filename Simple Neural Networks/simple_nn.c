#include "simple_nn.h"


double single_in_single_out (double input, double weight){
    return(input * weight);
}

double multiple_in_single_out(double* input, double* weight, int length){
    double predicted_value;

    predicted_value = weighted_sum(input, weight, length);
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

void multiple_in_multiple_out(double *input_vector, int INPUT_LEN, double *output_vector, int OUTPUT_LEN, double weight_matrix[INPUT_LEN][OUTPUT_LEN]){

}
