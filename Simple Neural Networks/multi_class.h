#ifndef MULTI_CLASS_H
#define MULTI_CLASS_H

#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define MULTICLASS_LAYER

#include "simple_nn.h"

#ifdef MULTICLASS_LAYER

// Loss function types for multi-class classification
typedef enum {
    CROSS_ENTROPY
} LossFunction;

// Derivative function types for multi-class layers
typedef enum {
    SOFTMAX_P
} Derivative;

// Layer structure for multi-class classification
typedef struct {
    int input_size;            // Number of inputs for the layer
    int output_size;           // Number of outputs for the layer (neurons)
    double **weights;          // Weight matrix for the layer
    double *biases;            // Bias vector for the layer
    double *input_vector;      // Pointer to the input vector for the layer
    double *output_vector;     // Pointer to the output vector for the layer
    Activation activation;      // Activation function used in the layer
    Derivative derivative;      // Derivative function for backpropagation
    LossFunction loss_func;    // Loss function used for the layer
} Layer; 

// MultiClassNeuralNetwork structure
typedef struct {
    int num_layers;            // Total number of layers in the network
    Layer *layers;             // Pointer to an array of layers
    double *output_vector;     // Final output of the network
    double learning_rate;      // Learning rate for the network
} NeuralNetwork;

#endif // MULTICLASS_LAYER

// Backpropagation for multi-class classification
void backpropagation_multi_class(NeuralNetwork *nn, double *input_vector, double *expected_values, 
                                 double learning_rate, Optimizer *optimizer, Regularizer *regularizer);

// Softmax and Cross-Entropy Loss Functions for Multi-Class
void softmax(double *input_vector, double *output_vector, int length);
double cross_entropy_loss(double *predicted, double *actual, int size);
void compute_cross_entropy_derivative(double *predicted, double *actual, double *derivative_out, int size);

/**
 * Computes the softmax activation function for multi-class classification.
 * The softmax function converts logits (raw class scores) into class probabilities.
 * 
 * @param input_vector A pointer to the logits (raw class scores).
 * @param output_vector A pointer to the output probabilities (softmax output).
 * @param length The length of the input/output vector.
 */
void softmax(double *input_vector, double *output_vector, int length);

/**
 * Computes the cross-entropy loss function for multi-class classification.
 * Cross-entropy compares the predicted class probabilities to the actual class labels.
 * 
 * @param predicted A pointer to the predicted probabilities (softmax output).
 * @param actual A pointer to the one-hot encoded actual class labels.
 * @param size The number of classes (length of predicted and actual vectors).
 * 
 * @return The cross-entropy loss value.
 */
double cross_entropy_loss(double *predicted, double *actual, int size);

/**
 * Performs backpropagation for multi-class classification using cross-entropy loss
 * and softmax output. This function adjusts the weights of the neural network based on
 * the computed gradients from the softmax and cross-entropy loss.
 * 
 * @param nn A pointer to the neural network structure.
 * @param input_vector A pointer to the input vector.
 * @param expected_values A pointer to the expected (true) class labels in one-hot encoding.
 * @param learning_rate The learning rate for weight updates.
 * @param optimizer A pointer to the optimizer configuration.
 * @param regularizer A pointer to the regularizer configuration.
 */
void backpropagation_multi_class(NeuralNetwork *nn, double *input_vector, double *expected_values, 
                                 double learning_rate, Optimizer *optimizer, Regularizer *regularizer);

/**
 * Updates the weights of a neural network layer using the gradients computed during
 * backpropagation, for multi-class classification.
 * 
 * @param optimizer A pointer to the optimizer configuration (e.g., SGD, Adam).
 * @param weights A pointer to the weights matrix of the layer.
 * @param gradients A pointer to the gradient matrix for the weights.
 * @param length The number of weights in the matrix.
 * @param regularizer A pointer to the regularizer configuration.
 */
void update_weights_multi_class(Optimizer *optimizer, double *weights, double *gradients, 
                                int length, Regularizer *regularizer);

/**
 * Computes the derivative of the cross-entropy loss with respect to the softmax output.
 * This gradient is used during backpropagation to update the weights of the final layer.
 * 
 * @param predicted A pointer to the predicted probabilities (softmax output).
 * @param actual A pointer to the actual class labels (one-hot encoded).
 * @param derivative_out A pointer to the output gradient vector.
 * @param size The number of classes.
 */
void compute_cross_entropy_derivative(double *predicted, double *actual, double *derivative_out, int size);

/**
 * Computes the classification accuracy by comparing the predicted probabilities
 * to the true labels (one-hot encoded). The accuracy is the proportion of correct
 * predictions.
 * 
 * @param predicted A pointer to the predicted probabilities (softmax output).
 * @param actual A pointer to the actual class labels (one-hot encoded).
 * @param size The number of classes.
 * 
 * @return The accuracy as a double value (between 0 and 1).
 */
double compute_accuracy(double *predicted, double *actual, int size);

/**
 * Computes the precision for a specific class in a multi-class classification task.
 * Precision is defined as the number of true positives divided by the sum of true 
 * positives and false positives for a given class.
 * 
 * @param predicted A pointer to the predicted class probabilities (softmax output).
 * @param actual A pointer to the actual class labels (one-hot encoded).
 * @param class_index The index of the class to compute precision for.
 * @param size The number of samples in the batch.
 * 
 * @return The precision for the specified class.
 */
double compute_precision(double *predicted, double *actual, int class_index, int size);

/**
 * Computes the recall for a specific class in a multi-class classification task.
 * Recall is defined as the number of true positives divided by the sum of true 
 * positives and false negatives for a given class.
 * 
 * @param predicted A pointer to the predicted class probabilities (softmax output).
 * @param actual A pointer to the actual class labels (one-hot encoded).
 * @param class_index The index of the class to compute recall for.
 * @param size The number of samples in the batch.
 * 
 * @return The recall for the specified class.
 */
double compute_recall(double *predicted, double *actual, int class_index, int size);

/**
 * Computes the confusion matrix for a multi-class classification task. The confusion matrix
 * provides insight into which classes are being correctly predicted and where misclassifications
 * are occurring.
 * 
 * @param predicted A pointer to the predicted class probabilities (softmax output).
 * @param actual A pointer to the actual class labels (one-hot encoded).
 * @param num_classes The number of classes in the classification task.
 * @param size The number of samples in the batch.
 * @param confusion_matrix A pointer to the output confusion matrix (must be preallocated).
 */
void compute_confusion_matrix(double *predicted, double *actual, int num_classes, int size, int **confusion_matrix);

/**
 * Converts the predicted softmax probabilities into a class label by selecting the 
 * index with the highest probability. This is typically done after computing the 
 * softmax output to map probabilities to class labels.
 * 
 * @param predicted A pointer to the predicted probabilities (softmax output).
 * @param num_classes The number of classes in the classification task.
 * 
 * @return The index of the class with the highest probability.
 */
int predicted_class(double *predicted, int num_classes);

/**
 * Helper function to convert a one-hot encoded vector to the class index.
 * The one-hot encoding represents the true label where only one element is 1 
 * (representing the class index), and the rest are 0.
 * 
 * @param actual A pointer to the actual class labels (one-hot encoded).
 * @param num_classes The number of classes in the classification task.
 * 
 * @return The index of the class that is 1 in the one-hot encoded vector.
 */
int actual_class(double *actual, int num_classes);


#endif // MULTI_CLASS_NN_H
