#ifndef SIMPLE_NN_H
#define SIMPLE_NN_H

#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <limits.h>
 
#include "lin_alg.h"

static double *momentum_velocity = NULL; // Velocity for momentum-based methods
static double *m_t = NULL; // First moment estimate for Adam
static double *v_t = NULL; // Second moment estimate for Adam

// Enum for learning rate scheduling
typedef enum {
    CONSTANT,
    STEP_DECAY,
    EXPONENTIAL_DECAY
} LearningRateSchedule;

// Learning rate structure
typedef struct {
    LearningRateSchedule schedule;
    double initial_lr;
    int step_size;
    double decay_rate;
} LearningRate;

// Enum for regularization types
typedef enum {
    NONE,
    L1,
    L2
} RegularizationType;

// Regularization structure
typedef struct {
    RegularizationType reg_type;
    double lambda; // Regularization strength
} Regularizer;

// Loss function types
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

// Optimizer types
typedef enum {
    SGD,            // Stochastic Gradient Descent
    MOMENTUM,       // Gradient Descent with Momentum
    ADAM            // Adaptive Moment Estimation
} OptimizerType;

// Optimizer structure
typedef struct {
    OptimizerType type;
    double learning_rate;
    double momentum; // For momentum-based optimizers
    double beta1;    // For Adam optimizer
    double beta2;    // For Adam optimizer
    double epsilon;  // Small constant to avoid division by zero
} Optimizer;

// Activation function types
typedef enum {
    RELU,
    SIGMOID,
    SOFTMAX,
    NO_ACTIVATION 
} Activation;

// Layer structure to encapsulate the weights, biases, and activation function
typedef struct {
    int input_size;            // Number of inputs for the layer
    int output_size;           // Number of outputs for the layer (neurons)
    double **weights;          // Weight matrix for the layer
    double *biases;            // Bias vector for the layer
    double *input_vector;      // Pointer to the input vector for the layer
    double *output_vector;     // Pointer to the output vector for the layer
    Activation activation;      // Activation function used in the layer
    Derivative derivative;      // Derivative function used in backpropagation
    LossFunction loss_func;     // Loss function used for the layer
} Layer;

typedef struct {
    int num_layers;            // Total number of layers in the network
    Layer *layers;             // Pointer to an array of layers
    double *output_vector;     // Final output of the network
    double learning_rate;      // Learning rate for the network
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

// Neural network learning
void bruteforce_learning(double *input_vector, double *expected_values, double learning_rate, uint32_t iterations, Layer *layer);
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

// Derivative of activation functions
void sigmoid_derivative(double *input_vector, double *output_vector, int length);
void relu_derivative(double *input_vector, double *output_vector, int length);
void softmax_derivative(double *input_vector, double *output_vector, int length);

// Loss and loss derivative
double compute_loss(LossFunction loss_function, double *predicted, double *actual, int size);
void compute_loss_derivative(LossFunction loss_function, double *predicted, double *actual, double *derivative_out, int size);

// Deep neural network forward pass
void deep_nn(double *input_vector, int input_size, 
             double *output_vector, int output_size, 
             Layer *layers, int num_layers);

// Optimization 
void update_weights(Optimizer *optimizer, double *weights, double *gradients, int length, Regularizer *regularizer);
Optimizer create_optimizer(OptimizerType type, double learning_rate, double momentum, double beta1, double beta2, double epsilon);


static void apply_activation(double *output_vector, int size, Activation activation);
static void apply_derivative(double *output_vector, int size, Derivative derivative);

#endif
