#include "multi_class.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

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
    
    // Find the maximum value for numerical stability
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

/**
 * @brief Computes the cross-entropy loss.
 *
 * The cross-entropy loss is commonly used in multi-class classification problems.
 * It measures the difference between the predicted probabilities (softmax output) 
 * and the actual one-hot encoded class labels.
 *
 * @param predicted A pointer to the predicted probabilities (softmax output).
 * @param actual A pointer to the one-hot encoded actual class labels.
 * @param size The number of classes.
 * 
 * @return The cross-entropy loss value.
 */
double cross_entropy_loss(double *predicted, double *actual, int size) {
    double loss = 0.0;
    for (int i = 0; i < size; i++) {
        if (actual[i] > 0) { // Avoid log(0)
            loss -= actual[i] * log(predicted[i]);
        }
    }
    return loss;
}

/**
 * @brief Performs backpropagation for multi-class classification.
 *
 * This function propagates the error backward through the neural network
 * and updates the weights based on the gradients computed from the cross-entropy
 * loss and softmax output.
 *
 * @param nn A pointer to the neural network structure.
 * @param input_vector A pointer to the input vector.
 * @param expected_values A pointer to the expected (true) class labels in one-hot encoding.
 * @param learning_rate The learning rate for weight updates.
 * @param optimizer A pointer to the optimizer configuration.
 * @param regularizer A pointer to the regularizer configuration.
 */
void backpropagation_multi_class(NeuralNetwork *nn, double *input_vector, double *expected_values, 
                                 double learning_rate, Optimizer *optimizer, Regularizer *regularizer) {
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
        double *weight_gradient = (double *)malloc(current_layer->output_size * current_layer->input_size * sizeof(double));
        update_weights_multi_class(optimizer, current_layer->weights, weight_gradient, current_layer->input_size * current_layer->output_size, regularizer);

        // Update weights and biases
        for (int i = 0; i < current_layer->output_size; i++) {
            for (int j = 0; j < current_layer->input_size; j++) {
                current_layer->weights[i * current_layer->input_size + j] -= learning_rate * weight_gradient[i * current_layer->input_size + j];
            }
            current_layer->biases[i] -= learning_rate * loss_gradient[i];
        }
        free(weight_gradient);
    }
    free(predicted_output);
    free(loss_gradient);
}

/**
 * @brief Computes the accuracy for multi-class classification.
 *
 * This function compares the predicted class probabilities to the true one-hot
 * encoded labels and returns the proportion of correct predictions.
 *
 * @param predicted A pointer to the predicted probabilities (softmax output).
 * @param actual A pointer to the actual class labels (one-hot encoded).
 * @param size The number of classes.
 * 
 * @return The accuracy as a double value.
 */
double compute_accuracy(double *predicted, double *actual, int size) {
    int predicted_class_idx = 0;
    int actual_class_idx = 0;
    
    // Find the index of the maximum predicted probability (the predicted class)
    for (int i = 1; i < size; i++) {
        if (predicted[i] > predicted[predicted_class_idx]) {
            predicted_class_idx = i;
        }
        if (actual[i] > actual[actual_class_idx]) {
            actual_class_idx = i;
        }
    }

    // Return 1 if the predicted class is the actual class, otherwise 0
    return (predicted_class_idx == actual_class_idx) ? 1.0 : 0.0;
}

/**
 * @brief Converts predicted softmax output to a one-hot encoded vector.
 *
 * This function takes the predicted probabilities and sets the maximum probability
 * index to 1, while all other indices are set to 0, effectively producing a one-hot encoded vector.
 *
 * @param predicted A pointer to the predicted probabilities (softmax output).
 * @param output A pointer to the one-hot encoded output vector.
 * @param size The number of classes.
 */
void convert_to_one_hot(double *predicted, double *output, int size) {
    int max_index = 0;
    for (int i = 1; i < size; i++) {
        if (predicted[i] > predicted[max_index]) {
            max_index = i;
        }
    }

    // Set the max_index to 1, all others to 0
    for (int i = 0; i < size; i++) {
        output[i] = (i == max_index) ? 1.0 : 0.0;
    }
}

/**
 * @brief Displays the predicted class probabilities for a multi-class classification problem.
 *
 * This function prints the predicted probabilities (softmax output) for each class.
 *
 * @param predicted A pointer to the predicted probabilities (softmax output).
 * @param size The number of classes.
 */
void display_predicted_probabilities(double *predicted, int size) {
    printf("Predicted class probabilities:\n");
    for (int i = 0; i < size; i++) {
        printf("Class %d: %f\n", i, predicted[i]);
    }
}
