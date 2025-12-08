#ifndef MLP_H
#define MLP_H

#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "../Simple Neural Networks/simple_nn.h"

#define Layer MLP_Layer

typedef struct Layer {
    int input_size;             // Number of incoming features
    int output_size;            // Number of neurons in the layer

    double **weights;           // Weight matrix W[j][i]
    double *biases;             // Bias vector b[j]

    double *input_vector;       // Cached input used during backpropagation
    double *output_vector;      // Forward activation output

    Activation activation;      // Nonlinear activation applied to this layer
    Derivative derivative;      // Derivative used during gradient computation
    LossFunction loss;          // Loss function for the output layer
} Layer;

typedef struct {
    int num_layers;             // Total number of layers in the network
    Layer *layers;              // Array of layers defining the MLP

    double *output_vector;      // Final network output after forward propagation
    double learning_rate;       // Base learning rate for optimization

    Optimizer optimizer;        // Optimization strategy (SGD, Momentum, Adam)
    Regularizer regularizer;    // Regularization strategy (L1, L2)
} MLP;

// Creates an MLP by specifying layer sizes and corresponding activations
MLP create_mlp(int *layer_sizes, int num_layers, Activation *activations);

// Creates a single fully connected layer with randomly initialized weights
Layer create_mlp_layer(int input_size, int output_size, Activation activation);

// Releases memory used by all layers composing the network
void destroy_mlp(MLP *mlp);

// Releases memory of a single layer
void destroy_mlp_layer(Layer *layer);

// Computes network output by propagating inputs through successive layers
void mlp_forward(MLP *mlp, double *input_vector);

// Computes gradients for all parameters using target values and updates weights
void mlp_backprop(MLP *mlp, double *input_vector, double *expected_output);

#undef Layer

#endif
