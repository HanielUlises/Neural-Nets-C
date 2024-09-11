#ifndef SIMPLE_NN_H
#define SIMPLE_NN_H

#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

// Constants for the neural network
#define NUM_FEATURES 3
#define NUM_EXAMPLES 3
#define OUTPUTS 3
#define HIDDEN_SIZE 3
#define INPUT_SIZE 3
#define OUTPUT_SIZE 3
#define NUM_OF_HID_NODES 3
#define NUM_OF_OUT_NODES 1

// Activation function type for flexibility
typedef enum {
    RELU,
    SIGMOID,
    SOFTMAX,
    NONE // For layers without activation
} Activation;

// Layer structure to encapsulate the weights, biases, and activation function
typedef struct {
    int input_size;      // Number of inputs for the layer
    int output_size;     // Number of outputs for the layer (neurons)
    double **weights;    // Weight matrix for the layer
    double *biases;      // Bias vector for the layer
    Activation activation; // Activation function used in the layer
} Layer;

// Neural Network structure, contains an array of layers
typedef struct {
    int num_layers;   // Total number of layers in the network
    Layer *layers;    // Pointer to an array of layers
    double *output_vector; // Final output of the network
} NeuralNetwork;

// Neural network layer creation
Layer create_layer(int input_size, int output_size, Activation activation);

// Constructor
NeuralNetwork create_neural_network(int num_layers, int *layer_sizes, Activation *activations);

// Perform forward pass through the neural network
void forward_pass(NeuralNetwork *nn, double *input_vector);

// Multiply a single input by a weight to produce a single output
double single_in_single_out(double input, double weight);

// Weighted sum of multiple inputs against corresponding weights
double multiple_in_single_out(double* input, double* weight, int length);

// Single scalar input producing multiple outputs
void single_in_multiple_out(double scalar, double* w_vect, double* out_vect, int length);

// Multiple inputs producing multiple outputs
void multiple_in_multiple_out(double *input_vector, int INPUT_LEN, double *output_vector, int OUTPUT_LEN, double **weight_matrix);

// Neural network hidden layer transformation
void hidden_layer_nn(double *input_vector, Layer *hidden_layer, Layer *output_layer, double *output_vector);

// Utility functions for operations like matrix multiplication and error calculations
double weighted_sum(double *input, double *weight, int length);
void matrix_vector_multiplication(double *input_vector, int INPUT_LEN, double *output_vector, int OUTPUT_LEN, double **weight_matrix);

// Error calculation functions
double find_error(double input, double weight, double expected_value);
double find_error_simple(double yhat, double y);

// Brute force learning for optimizing weights
void bruteforce_learning(double input, double weight, double expected_values, double step_amount, uint32_t itr);

// Data normalization
void normalize_data(double *input_vector, double *output_vector, int LEN);
void normalize_data_2D(int row, int col, double **input_matrix, double **output);

// Random weight initialization for 1D and 2D arrays
void random_weight_initialization(int HIDDEN_LENGTH, int INPUT_LENGTH, double **weights_matrix);
void random_weight_init_1D(double *output_vector, uint32_t LEN);

// Activation functions
void softmax(double *input_vector, double *output_vector, int length);
void relu(double *input_vector, double *output_vector, int length);
void sigmoid(double *input_vector, double *output_vector, int length);

// Deep neural network forward pass
void deep_nn(double *input_vector, int input_size, 
             double *output_vector, int output_size, 
             Layer *layers, int num_layers);


#endif
