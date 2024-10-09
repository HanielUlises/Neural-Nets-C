#include "multi_class.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Function to compute the softmax activation
/**
 * @brief Computes the softmax activation for a given input vector.
 *
 * The softmax function converts a vector of raw scores (logits) into probabilities 
 * that sum up to 1. It is commonly used in multi-class classification problems.
 *
 * The softmax function is defined mathematically as:
 *
 *     softmax(z_i) = exp(z_i) / Σ(exp(z_j)) for j = 1 to K
 *
 * Where:
 * - z_i is the i-th element of the input vector,
 * - exp(z_i) is the exponential of z_i,
 * - Σ(exp(z_j)) is the sum of the exponentials of all elements in the input vector,
 * - K is the total number of elements in the input vector.
 *
 * The subtraction of the maximum value of the input vector before exponentiation 
 * is done for numerical stability to prevent overflow:
 *
 *     softmax(z_i) = exp(z_i - max(z)) / Σ(exp(z_j - max(z)))
 *
 * This ensures that the largest exponent is 0, making the calculations more stable.
 *
 * @param input_vector A pointer to the input data vector (logits).
 * @param output_vector A pointer to the vector where the probabilities will be stored.
 * @param length The number of elements in the input vector.
 */
void softmax(double *input_vector, double *output_vector, int length) {
    double max_val = input_vector[0];
    
    // We ind the maximum value for numerical stability
    for (int i = 1; i < length; i++) {
        if (input_vector[i] > max_val) {
            max_val = input_vector[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        output_vector[i] = exp(input_vector[i] - max_val);
        sum += output_vector[i];
    }

    // Normalize to get probabilities
    for (int i = 0; i < length; i++) {
        output_vector[i] /= sum;
    }
}

// Function to compute cross-entropy loss
double cross_entropy_loss(double *predicted, double *actual, int size) {
    double loss = 0.0;
    for (int i = 0; i < size; i++) {
        if (actual[i] > 0) { // Avoid log(0)
            loss -= actual[i] * log(predicted[i]);
        }
    }
    return loss;
}

// Backpropagation for multi-class classification using cross-entropy loss
void backpropagation_multi_class(NeuralNetwork *nn, double *input_vector, double *expected_values, 
                                 double learning_rate, Optimizer *optimizer, Regularizer *regularizer) {
    // Perform forward pass to compute outputs
    double *predicted_output = (double *)malloc(nn->layers[nn->num_layers - 1].output_size * sizeof(double));
    deep_nn(input_vector, nn->layers[0].input_size, predicted_output, nn->layers[nn->num_layers - 1].output_size, nn->layers, nn->num_layers);

    // Compute the loss
    double loss = cross_entropy_loss(predicted_output, expected_values, nn->layers[nn->num_layers - 1].output_size);
    printf("Loss: %f\n", loss);

    // Compute the gradient of the loss with respect to the output
    double *loss_gradient = (double *)malloc(nn->layers[nn->num_layers - 1].output_size * sizeof(double));
    compute_cross_entropy_derivative(predicted_output, expected_values, loss_gradient, nn->layers[nn->num_layers - 1].output_size);

    // Backpropagate the error through each layer
    for (int layer_idx = nn->num_layers - 1; layer_idx >= 0; layer_idx--) {
        Layer *current_layer = &nn->layers[layer_idx];
        double *prev_layer_output = (layer_idx == 0) ? input_vector : nn->layers[layer_idx - 1].output_vector;
        
        // Calculate the gradients for weights and biases
        double *gradients = (double *)malloc(current_layer->output_size * current_layer->input_size * sizeof(double));
        for (int j = 0; j < current_layer->output_size; j++) {
            for (int k = 0; k < current_layer->input_size; k++) {
                gradients[j * current_layer->input_size + k] = loss_gradient[j] * prev_layer_output[k];
            }
        }

        // Update the weights and biases
        update_weights_multi_class(optimizer, current_layer->weights[0], gradients, current_layer->input_size * current_layer->output_size, regularizer);

        // Update the biases (simple gradient descent)
        for (int j = 0; j < current_layer->output_size; j++) {
            current_layer->biases[j] -= learning_rate * loss_gradient[j]; 
        }

        // Calculate loss gradient for the previous layer if not the input layer
        if (layer_idx > 0) {
            double *next_layer_gradient = (double *)malloc(current_layer->input_size * sizeof(double));
            for (int j = 0; j < current_layer->input_size; j++) {
                next_layer_gradient[j] = 0.0;
                for (int k = 0; k < current_layer->output_size; k++) {
                     // Sum of gradients
                    next_layer_gradient[j] += loss_gradient[k] * current_layer->weights[k][j];
                }
            }

            // Apply the derivative of the activation function for the current layer
            apply_derivative(next_layer_gradient, current_layer->input_size, current_layer->derivative);

            // Copy the next layer's gradient to the loss gradient for the next iteration
            free(loss_gradient);
            loss_gradient = next_layer_gradient;
        } else {
            free(loss_gradient);
        }
        free(gradients);
    }

    free(predicted_output);
}

// Update weights for multi-class classification
void update_weights_multi_class(Optimizer *optimizer, double *weights, double *gradients, 
                                int length, Regularizer *regularizer) {
    // Update weights based on optimizer
    if (optimizer->type == SGD) {
        for (int i = 0; i < length; i++) {
            weights[i] -= optimizer->learning_rate * gradients[i]; // Simple SGD update
        }
    } else if (optimizer->type == ADAM) {
        // Pending
    }
    // Handle regularization if applicable
    if (regularizer->reg_type == L2) {
        for (int i = 0; i < length; i++) {
            // L2 regularization
            weights[i] -= optimizer->learning_rate * regularizer->lambda * weights[i]; 
        }
    }
}

// Compute the derivative of cross-entropy loss
void compute_cross_entropy_derivative(double *predicted, double *actual, double *derivative_out, int size) {
    for (int i = 0; i < size; i++) {
        // dL/dy = y_pred - y_true
        derivative_out[i] = predicted[i] - actual[i]; 
    }
}
