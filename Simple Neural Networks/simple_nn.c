#include "simple_nn.h"
#include <stddef.h>

double single_in_single_out(double input, double weight) {
    return input * weight;
}

double multiple_in_single_out(double* input, double* weight, int length) {
    return weighted_sum(input, weight, length);
}

void single_in_multiple_out(double scalar, double* w_vect, double* out_vect, int length) {
    element_wise_multiply(scalar, w_vect, out_vect, length);
}

void multiple_in_multiple_out(double *input_vector, int INPUT_LEN, double *output_vector, int OUTPUT_LEN, double weight_matrix[][OUTPUTS]) {
    for (int i = 0; i < OUTPUT_LEN; i++) {
        output_vector[i] = 0;
        for (int j = 0; j < INPUT_LEN; j++) {
            output_vector[i] += input_vector[j] * weight_matrix[j][i];
        }
    }
}

void hidden_layer_nn(double *input_vector, double in_to_hid_weights[HIDDEN_SIZE][INPUT_SIZE], double hid_to_out_weights[OUTPUT_SIZE][HIDDEN_SIZE], double *output_vector){
    double hidden_pred_vector[HIDDEN_SIZE];
    matrix_vector_multiplication(input_vector, INPUT_SIZE, hidden_pred_vector, HIDDEN_SIZE, in_to_hid_weights);
    matrix_vector_multiplication(hidden_pred_vector, HIDDEN_SIZE, output_vector, OUTPUT_SIZE, output_vector);
}

void matrix_vector_multiplication (double *input_vector, int INPUT_LEN, double *output_vector, int OUTPUT_LEN, double weight_matrix[][OUTPUTS]) {
    multiple_in_multiple_out(input_vector, INPUT_LEN, output_vector, OUTPUT_LEN, weight_matrix);
}

double weighted_sum(double* input, double* weight, int length) {
    double output = 0.0;

    for (size_t i = 0; i < length; i++) {
        output += input[i] * weight[i];
    }

    return output;
}

void element_wise_multiply(double input_scalar, double* weight_vector, double* output_vector, int length) {
    for (size_t i = 0; i < length; i++) {
        output_vector[i] = input_scalar * weight_vector[i];
    }
}

double finding_error(double input, double weight, double expected_value){
    return pow(((input *weight) - expected_value), 2);
}

double find_error_simple(double yhat, double y){
    return pow((yhat - y), 2);
}