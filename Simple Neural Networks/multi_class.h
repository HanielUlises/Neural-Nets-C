#ifndef MULTI_CLASS_H
#define MULTI_CLASS_H

#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "simple_nn.h"  

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

#endif // MULTI_CLASS_NN_H
