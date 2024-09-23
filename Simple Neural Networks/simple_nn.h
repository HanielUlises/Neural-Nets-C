#pragma once

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

static double *momentum_velocity = NULL; // Velocity for momentum-based methods
static double *m_t = NULL; // First moment estimate for Adam
static double *v_t = NULL; // Second moment estimate for Adam

typedef enum {
    CONSTANT,
    STEP_DECAY,
    EXPONENTIAL_DECAY
} LearningRateSchedule;

typedef struct {
    LearningRateSchedule schedule;
    double initial_lr;
    int step_size;
    double decay_rate;
} LearningRate;

typedef enum {
    NONE,
    L1,
    L2
} RegularizationType;

typedef struct {
    RegularizationType reg_type;
    double lambda; // Regularization strength
} Regularizer;

// Loss Functions
typedef enum {
    MEAN_SQUARED_ERROR,
    CROSS_ENTROPY
} LossFunction;

// Derivative function types
typedef enum {
    RELU_P,
    SIGMOID_P,
    SOFTMAX_P,
    NO_DERIVATIVE
} Derivative;

typedef enum {
    SGD,            // Stochastic Gradient Descent
    MOMENTUM,       // Gradient Descent with Momentum
    ADAM            // Adaptive Moment Estimation
} OptimizerType;

typedef struct {
    OptimizerType type;
    double learning_rate;
    double momentum; // For momentum-based optimizers
    double beta1;    // For Adam optimizer
    double beta2;    // For Adam optimizer
    double epsilon;  // Small constant to avoid division by zero
} Optimizer;

// Activation function type for flexibility
typedef enum {
    RELU,
    SIGMOID,
    SOFTMAX,
    NO_ACTIVATION 
} Activation;

// Layer structure to encapsulate the weights, biases, and activation function
typedef struct {
    int input_size;        // Number of inputs for the layer
    int output_size;       // Number of outputs for the layer (neurons)
    double **weights;      // Weight matrix for the layer
    double *biases;        // Bias vector for the layer
    Activation activation; // Activation function used in the layer
    Derivative derivative; // Derivative function used in backpropagation
} Layer;

// Neural Network structure, contains an array of layers
typedef struct {
    int num_layers;        // Total number of layers in the network
    Layer *layers;         // Pointer to an array of layers
    double *output_vector; // Final output of the network
} NeuralNetwork;

// Neural network layer creation
Layer create_layer(int input_size, int output_size, Activation activation);

// Constructor
NeuralNetwork create_neural_network(int num_layers, int *layer_sizes, Activation *activations);

// Destructors
void destroy_layer(Layer *layer);
void destroy_neural_network(NeuralNetwork *nn);

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

// Element-wise multiplication of a scalar with each element in a vector.
void element_wise_multiply(double input_scalar, double* weight_vector, double* output_vector, int length);

// Neural network hidden layer transformation
void hidden_layer_nn(double *input_vector, Layer *hidden_layer, Layer *output_layer, double *output_vector);

// Utility functions for operations like matrix multiplication and error calculations
double weighted_sum(double *input, double *weight, int length);
void matrix_vector_multiplication(double *input_vector, int INPUT_LEN, double *output_vector, int OUTPUT_LEN, double **weight_matrix);

// Error calculation functions
double find_error(double input, double weight, double expected_value);
double find_error_simple(double yhat, double y);

void apply_activation(double *output_vector, int size, Activation activation);

// Neural network learning
/// @param input represents the x-ith input from the X input set of the neural network
/// @param weight represents the w-ith input from the W input set of the neural network
/// @param expected_values represent the y actual value that we expect
/// @param stemp_ampunt learning rate 
/// @param itr mu-th element 

// i) Brute force learning for optimizing weights
void bruteforce_learning(double *input_vector, double *expected_values, double learning_rate, uint32_t iterations, Layer *layer);
// ii) Gradient descent learning for optimizing weights
void gradient_descent(double *input_vector, double *expected_values, double learning_rate, uint32_t iterations, Layer *layer, LossFunction loss_function);

// Backpropagation to compute gradients and adjust weights
void backpropagation(NeuralNetwork *nn, double *input_vector, double *expected_values, double learning_rate);

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

// Derivative of activation function
void sigmoid_derivative(double *input_vector, double *output_vector, int length);
void relu_derivative(double *input_vector, double *output_vector, int length);
void softmax_derivative(double *input_vector, double *output_vector, int length);

// listing
void apply_derivative(double *output_vector, int size, Derivative derivative);

double compute_loss(LossFunction loss_function, double *predicted, double *actual, int size);
void compute_loss_derivative(LossFunction loss_function, double *predicted, double *actual, double *derivative_out, int size);

// Deep neural network forward pass
void deep_nn(double *input_vector, int input_size, 
             double *output_vector, int output_size, 
             Layer *layers, int num_layers);

// Optimization 
void update_weights(Optimizer *optimizer, double *weights, double *gradients, int length);
Optimizer create_optimizer(OptimizerType type, double learning_rate, double momentum, double beta1, double beta2, double epsilon);

#endif
