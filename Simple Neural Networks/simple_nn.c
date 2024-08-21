#include "simple_nn.h"

// Multiply a single input by a weight to produce a single output.
double single_in_single_out(double input, double weight) {
    return input * weight;
}

// Weighted sum of multiple inputs against corresponding weights.
double multiple_in_single_out(double* input, double* weight, int length) {
    return weighted_sum(input, weight, length);
}

void single_in_multiple_out(double scalar, double* w_vect, double* out_vect, int length) {
    element_wise_multiply(scalar, w_vect, out_vect, length);
}

// Computes the output vector from an input vector and a matrix of weights.
void multiple_in_multiple_out(double *input_vector, int INPUT_LEN, double *output_vector, int OUTPUT_LEN, double weight_matrix[][OUTPUTS]) {
    for (int i = 0; i < OUTPUT_LEN; i++) {
        output_vector[i] = 0;
        for (int j = 0; j < INPUT_LEN; j++) {
            output_vector[i] += input_vector[j] * weight_matrix[j][i];
        }
    }
}

// A neural network hidden layer transformation, taking input through two layers of weights.
void hidden_layer_nn(double *input_vector, double in_to_hid_weights[HIDDEN_SIZE][INPUT_SIZE], double hid_to_out_weights[OUTPUT_SIZE][HIDDEN_SIZE], double *output_vector){
    double hidden_pred_vector[HIDDEN_SIZE];
    matrix_vector_multiplication(input_vector, INPUT_SIZE, hidden_pred_vector, HIDDEN_SIZE, in_to_hid_weights);
    matrix_vector_multiplication(hidden_pred_vector, HIDDEN_SIZE, output_vector, OUTPUT_SIZE, output_vector);
}

// Matrix-vector multiplication used in neural networks.
void matrix_vector_multiplication (double *input_vector, int INPUT_LEN, double *output_vector, int OUTPUT_LEN, double weight_matrix[][OUTPUTS]) {
    multiple_in_multiple_out(input_vector, INPUT_LEN, output_vector, OUTPUT_LEN, weight_matrix);
}

// Weighted sum of an array of inputs with an array of weights.
double weighted_sum(double* input, double* weight, int length) {
    double output = 0.0;

    for (size_t i = 0; i < length; i++) {
        output += input[i] * weight[i];
    }

    return output;
}

// Element-wise multiplication of a scalar with each element in a vector.
void element_wise_multiply(double input_scalar, double* weight_vector, double* output_vector, int length) {
    for (size_t i = 0; i < length; i++) {
        output_vector[i] = input_scalar * weight_vector[i];
    }
}

// Calculates squared error of a prediction based on input and expected value.
double find_error(double input, double weight, double expected_value){
    return pow(((input *weight) - expected_value), 2);
}

// Simple squared error between prediction and actual value.
double find_error_simple(double yhat, double y){
    return pow((yhat - y), 2);
}

/**
 * Perform a brute force learning algorithm to find a better weight for a single input feature.
 * This function adjusts the weight by iteratively increasing or decreasing it by a small step
 * and comparing the squared error between the predicted and expected values. The weight is adjusted
 * towards the direction that minimizes the error. This simplistic approach is a form of gradient descent,
 * although it does not compute the gradient explicitly.
 * 
 * @param input The input value to the neural network node.
 * @param weight The initial weight applied to the input.
 * @param expected_values The expected output value for the input.
 * @param step_amount The incremental step to adjust the weight by during each iteration.
 * @param itr The number of iterations to perform for adjusting the weight.
 * 
 * @note This function is designed for educational or simple scenarios where derivative-based
 * optimization is not feasible. It is not suitable for large-scale or complex neural network training!!
 */

void bruteforce_learning(double input, double weight, double expected_values, double step_amount, uint32_t itr){
    double prediction, error;
    double up_prediction, up_error;
    double down_prediction, down_error;

    for(int i = 0; i < itr; i++){
        prediction = input * weight;
        error = pow((prediction - expected_values), 2);
        printf("Error: %f Prediction: %f \r\n", error, prediction);

        up_prediction = input * (weight + step_amount);
        up_error = pow((expected_values - up_prediction), 2);

        down_prediction = input * (weight - step_amount);
        down_error = pow((expected_values - down_prediction), 2);

        if(down_error < up_error){
            weight = weight - step_amount;
        }else if (down_error > up_error){
            weight = weight + step_amount;
        }
    }
}

void normalize_data (double *input_vector, double *output_vector, int LEN){
    // Find maximum value
    size_t i = 0;
    double max = input_vector[0];

    for(i = 0; i < LEN; i++){
        if(input_vector[i] > max){
            max = input_vector[i];
        }
    }

    // Data normalization
    for(i = 0; i < LEN; i++){
        output_vector[i] = input_vector[i] / max;
    }
}

void normalize_data_2D(int row, int col, double **input_matrix, double **output){
    double max = -DBL_MAX;

    for(size_t i = 0; i < row; i++){
        for(size_t j = 0; j < row; j++){
            if(input_matrix[i][j] > max){
                max = input_matrix[i][j];
            }
        }
    }

    for(size_t i = 0; i < row; i++){
        for(size_t j = 0; j < row; j++){
            if(input_matrix[i][j] > max){
                output[i][j] = input_matrix[i][j] / max;
            }
        }
    }
}

void random_weight_initialization(int HIDDEN_LENGTH, int INPUT_LENGTH, double **weights_matrix){
    srand(2);
    double d_rand;
    
    for(size_t i = 0; i < HIDDEN_LENGTH; i++){
        for(size_t j = 0; j < INPUT_LENGTH; j++){
            d_rand = (rand() % 10);
            d_rand = d_rand / 10;

            weights_matrix[i][j] = d_rand;
        }
    }
}

void random_weight_init_1D(double *output_vector, uint32_t LEN){
    double d_rand;
    srand(2);

    for(int x = 0; x < LEN; x++){
        d_rand = (rand() % 10);
        d_rand = d_rand / 10;
        output_vector[x] = d_rand;
    }
}