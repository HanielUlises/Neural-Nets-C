#ifndef TESTS_H
#define TESTS_H

#include "simple_nn.h"

// Test functions for different neural network configurations

// Single input Single output neural network test
void test_siso_nn();

// Single input Multiple output neural network test
void test_simo_nn();

// Multiple input Single output neural network test
void test_miso_nn();

// Multiple input Multiple output neural network test
void test_mimo_nn();

// Hidden layer neural network test
void test_hidden_layer_nn();

// Brute force learning test
void test_brute_force();

// Forward propagation example test
void forward_propagation();

#endif
