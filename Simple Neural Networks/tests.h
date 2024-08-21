#include "simple_nn.h"

#define SAD_IDX     0
#define SICK_IDX    1
#define ACTIVE_IDX  2

/*
    Problem is to determine whether a person is sad or not based on 
    the ambience

    Prediction: Person's mood.
*/

// Single input Single output
void test_siso_nn();
// Single input Multiple output
void test_simo_nn();
// Multiple input single output
void test_miso_nn();
// Muliple input Multiple output
void test_mimo_nn();
// Hidden layer neural network
void test_hidden_layer_nn();

void forward_propagation();