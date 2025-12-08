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
 * @brief Extracts the index of the active class from a one-hot vector.
 *
 * The function assumes a valid one-hot encoding where exactly one element
 * equals 1.0. If this assumption is violated, -1 is returned.
 *
 * @param actual        One-hot encoded label vector.
 * @param num_classes   Length of the vector.
 * @return Index of the class, or -1 for invalid encodings.
 */
int actual_class(double *actual, int num_classes) {
    for (int i = 0; i < num_classes; i++) {
        if (actual[i] == 1.0) return i;
    }
    return -1;
}

/**
 * @brief Returns the argmax index from a probability/logit vector.
 *
 * This is typically used after a softmax layer to obtain the predicted class.
 *
 * @param predicted     Probability or logit vector.
 * @param num_classes   Length of the vector.
 * @return Index of the maximal component.
 */
int predicted_class(double *predicted, int num_classes) {
    int max_index = 0;
    double max_value = predicted[0];

    for (int i = 1; i < num_classes; i++) {
        if (predicted[i] > max_value) {
            max_value = predicted[i];
            max_index = i;
        }
    }
    return max_index;
}

/**
 * @brief Constructs a confusion matrix over a batch.
 *
 * The predicted and actual arrays are assumed to be laid out in
 * row-major form: sample 0 occupies [0 … num_classes-1], sample 1
 * occupies [num_classes … 2*num_classes-1], etc.
 *
 * @param predicted          Predicted probabilities (size × num_classes).
 * @param actual             One-hot labels (size × num_classes).
 * @param num_classes        Number of classes.
 * @param size               Number of samples.
 * @param confusion_matrix   Pre-allocated matrix[num_classes][num_classes].
 */
void compute_confusion_matrix(double *predicted, double *actual,
                              int num_classes, int size,
                              int **confusion_matrix)
{
    // Matrix reset
    for (int i = 0; i < num_classes; i++)
        for (int j = 0; j < num_classes; j++)
            confusion_matrix[i][j] = 0;

    // Streaming accumulation of classification outcomes
    for (int s = 0; s < size; s++) {
        int p = predicted_class(&predicted[s * num_classes], num_classes);
        int a = actual_class(&actual[s * num_classes], num_classes);

        if (p >= 0 && a >= 0)
            confusion_matrix[a][p] += 1;
    }
}

/**
 * @brief Gradient of cross-entropy with softmax logits.
 *
 * For softmax-with-cross-entropy, the backward signal reduces to:
 *   dL/dz = predicted − actual
 * which is both computationally and numerically optimal.
 *
 * @param predicted        Predicted probabilities.
 * @param actual           One-hot encoded targets.
 * @param derivative_out   Output gradient vector.
 * @param size             Vector length.
 */
void compute_cross_entropy_derivative(double *predicted, double *actual,
                                      double *derivative_out, int size)
{
    for (int i = 0; i < size; i++)
        derivative_out[i] = predicted[i] - actual[i];
}

/**
 * @brief Adapter: update a layer's weights stored as double** using the existing update_weights helper.
 *
 * This routine flattens a row-major (weights[row][col]) representation into a contiguous
 * temporary buffer only when necessary, then delegates to the generic update_weights()
 * so optimizer logic (SGD/Adam/etc.) is centralized.
 *
 * @param optimizer  Optimizer configuration.
 * @param weights    Layer weights as double** (rows == out, cols == in).
 * @param gradients  Gradients as a flat buffer length == rows*cols.
 * @param rows       Number of output neurons (rows).
 * @param cols       Number of input neurons (cols).
 * @param regularizer Regularizer descriptor or NULL.
 */
void update_weights_multi_class(Optimizer *optimizer, double **weights, double *gradients,
                                int rows, int cols, Regularizer *regularizer)
{
    if (rows <= 0 || cols <= 0) return;
    int length = rows * cols;

    if (!weights || !weights[0]) {
        double *flat = (double *)malloc(sizeof(double) * length);
        if (!flat) {
            fprintf(stderr, "update_weights_multi_class: allocation failed\n");
            return;
        }
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                flat[r * cols + c] = weights[r] ? weights[r][c] : 0.0;

        update_weights(optimizer, flat, gradients, length, regularizer);

        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                if (weights[r]) weights[r][c] = flat[r * cols + c];

        free(flat);
        return;
    }

    int contiguous = 1;
    for (int r = 0; r < rows; ++r) {
        if (weights[r] != &weights[0][r * cols]) {
            contiguous = 0;
            break;
        }
    }

    if (contiguous) {
        double *flat = &weights[0][0];
        update_weights(optimizer, flat, gradients, length, regularizer);
        return;
    }

    /* Not contiguous: gather, update, scatter */
    double *flat = (double *)malloc(sizeof(double) * length);
    if (!flat) {
        fprintf(stderr, "update_weights_multi_class: allocation failed\n");
        return;
    }

    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            flat[r * cols + c] = weights[r] ? weights[r][c] : 0.0;

    update_weights(optimizer, flat, gradients, length, regularizer);

    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            if (weights[r]) weights[r][c] = flat[r * cols + c];

    free(flat);
}

/**
 * @brief Computes class-level precision across a batch.
 *
 * Precision measures how "pure" the predictions for the given class are:
 *     TP / (TP + FP)
 *
 * @param predicted     Predicted probabilities (size × num_classes).
 * @param actual        One-hot labels (size × num_classes).
 * @param class_index   Class for which precision is measured.
 * @param num_classes   Number of classes.
 * @param size          Number of samples.
 * @return Precision value in [0,1].
 */
double compute_precision(double *predicted, double *actual,
                         int class_index, int num_classes, int size)
{
    int tp = 0, fp = 0;

    for (int i = 0; i < size; i++) {
        int p = predicted_class(&predicted[i * num_classes], num_classes);
        int a = actual_class(&actual[i * num_classes], num_classes);

        if (p == class_index && a == class_index) tp++;
        if (p == class_index && a != class_index) fp++;
    }

    if (tp + fp == 0) return 0.0;
    return (double)tp / (tp + fp);
}

/**
 * @brief Computes class-level recall across a batch.
 *
 * Recall quantifies how comprehensively the class was recovered:
 *     TP / (TP + FN)
 *
 * @param predicted     Predicted probabilities (size × num_classes).
 * @param actual        One-hot labels (size × num_classes).
 * @param class_index   Class for which recall is measured.
 * @param num_classes   Number of classes.
 * @param size          Number of samples.
 * @return Recall value in [0,1].
 */
double compute_recall(double *predicted, double *actual,
                      int class_index, int num_classes, int size)
{
    int tp = 0, fn = 0;

    for (int i = 0; i < size; i++) {
        int p = predicted_class(&predicted[i * num_classes], num_classes);
        int a = actual_class(&actual[i * num_classes], num_classes);

        if (a == class_index && p == class_index) tp++;
        if (a == class_index && p != class_index) fn++;
    }

    if (tp + fn == 0) return 0.0;
    return (double)tp / (tp + fn);
}


/**
 * @brief Backpropagation for multi-class classification using softmax + cross-entropy.
 *
 * Assumptions:
 * - Final-layer activation is softmax and loss is cross-entropy => dL/dz = predicted - expected.
 * - Layer weights are stored as weights[row][col] where row indexes output neuron.
 * - deep_nn populates each layer's output_vector.
 *
 * This function computes gradients for a single example (extend to mini-batches by summing gradients).
 */
void backpropagation_multi_class(NeuralNetwork *nn, double *input_vector, double *expected_values,
                                 double learning_rate, Optimizer *optimizer, Regularizer *regularizer)
{
    if (!nn || nn->num_layers <= 0) return;

    int L = nn->num_layers;
    Layer *last = &nn->layers[L - 1];
    int output_size = last->output_size;

    // Forward: fill per-layer output_vector using provided deep_nn
    // deep_nn signature: deep_nn(input_vector, input_size, output_vector, output_size, layers, num_layers);
    deep_nn(input_vector, nn->layers[0].input_size, nn->output_vector, output_size, nn->layers, nn->num_layers);

    // Copy predicted probabilities from network output (assuming nn->output_vector holds final softmax)
    double *predicted = nn->output_vector;

    // Compute top-layer gradient: dL/dz = predicted - expected
    double *grad = (double *)malloc(sizeof(double) * output_size);
    if (!grad) {
        fprintf(stderr, "backprop: allocation failed\n");
        return;
    }
    compute_cross_entropy_derivative(predicted, expected_values, grad, output_size);

    // We'll iterate layers backward, keeping track of the gradient w.r.t. each layer's outputs.
    //   For layer l, grad has length = layers[l].output_size.
    //   prev_grad is gradient w.r.t. inputs to the current layer (size = input_size).
    for (int l = L - 1; l >= 0; --l) {
        Layer *layer = &nn->layers[l];
        int out_sz = layer->output_size;
        int in_sz = layer->input_size;

        // prev activation vector (input to current layer)
        double *act_in = (l == 0) ? input_vector : nn->layers[l - 1].output_vector;
        double *act_out = layer->output_vector; // post-activation outputs

        // weight gradients: dW[i,j] = grad[i] * act_in[j]  (i: output neuron, j: input index)
        double *weight_gradients = (double *)calloc(out_sz * in_sz, sizeof(double));
        double *bias_gradients = (double *)calloc(out_sz, sizeof(double));
        if (!weight_gradients || !bias_gradients) {
            fprintf(stderr, "backprop: allocation failed for gradients\n");
            free(weight_gradients);
            free(bias_gradients);
            free(grad);
            return;
        }

        for (int i = 0; i < out_sz; ++i) {
            bias_gradients[i] = grad[i];
            for (int j = 0; j < in_sz; ++j) {
                weight_gradients[i * in_sz + j] = grad[i] * act_in[j];
            }
        }

        update_weights_multi_class(optimizer, layer->weights, weight_gradients, out_sz, in_sz, regularizer);
        update_weights(optimizer, layer->biases, bias_gradients, out_sz, regularizer);

        // Gradient for previous layer (if any): prev_grad[j] = sum_i W[i][j] * grad[i] * activation_derivative_i
        if (l > 0) {
            double *prev_grad = (double *)calloc(in_sz, sizeof(double));
            if (!prev_grad) {
                fprintf(stderr, "backprop: allocation failed prev_grad\n");
                free(weight_gradients);
                free(bias_gradients);
                free(grad);
                return;
            }

            // compute activation derivative of current layer outputs (dact/dz)
            double *act_derivative = (double *)malloc(sizeof(double) * out_sz);
            if (!act_derivative) {
                fprintf(stderr, "backprop: allocation failed act_derivative\n");
                free(prev_grad);
                free(weight_gradients);
                free(bias_gradients);
                free(grad);
                return;
            }

            // apply_derivative modifies provided array in-place according to layer->derivative
            memcpy(act_derivative, act_out, sizeof(double) * out_sz);
            apply_derivative(act_derivative, out_sz, layer->derivative);

            for (int j = 0; j < in_sz; ++j) {
                double acc = 0.0;
                for (int i = 0; i < out_sz; ++i) {
                    acc += layer->weights[i][j] * grad[i] * act_derivative[i];
                }
                prev_grad[j] = acc;
            }

            // swap: free old grad and replace with prev_grad for next iteration
            free(grad);
            grad = prev_grad;

            free(act_derivative);
        } else {
            // no previous layer — finished backprop
            free(grad);
            grad = NULL;
        }

        free(weight_gradients);
        free(bias_gradients);
    }

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
 * @brief Batch accuracy: compares argmax of each sample.
 *
 * @param predicted  shape: batch_size × num_classes
 * @param actual     shape: batch_size × num_classes (one-hot)
 * @param num_classes
 * @param batch_size
 * @return fraction correct in [0,1]
 */
double compute_accuracy_batch(double *predicted, double *actual, int num_classes, int batch_size)
{
    int correct = 0;
    for (int s = 0; s < batch_size; ++s) {
        int p = predicted_class(&predicted[s * num_classes], num_classes);
        int a = actual_class(&actual[s * num_classes], num_classes);
        if (p >= 0 && a >= 0 && p == a) ++correct;
    }
    return (double)correct / (double)batch_size;
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
