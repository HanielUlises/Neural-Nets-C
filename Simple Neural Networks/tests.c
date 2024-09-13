/* #include "tests.h"
#include <stdio.h>
#include <stdlib.h>

// Test for Single Input Single Output neural network
void test_siso_nn() {
    double input = 5.0;
    double weight = -2.0;
    double result = single_in_single_out(input, weight);

    printf("Single Input Single Output Test:\n");
    printf("Input: %f, Weight: %f, Prediction: %f\n", input, weight, result);
}

// Test for Multiple Input Single Output neural network
void test_miso_nn() {
    double inputs[NUM_FEATURES] = {12, 23, 50};
    double weights[NUM_FEATURES] = {-2, 2, 1};
    double result = multiple_in_single_out(inputs, weights, NUM_FEATURES);

    printf("Multiple Input Single Output Test:\n");
    printf("Inputs: [%f, %f, %f], Weights: [%f, %f, %f], Prediction: %f\n",
           inputs[0], inputs[1], inputs[2],
           weights[0], weights[1], weights[2],
           result);
}

// Test for Single Input Multiple Output neural network
void test_simo_nn() {
    double scalar = 0.9;
    double weights[OUTPUTS] = {-20, 95, 201};
    double results[OUTPUTS];

    single_in_multiple_out(scalar, weights, results, OUTPUTS);

    printf("Single Input Multiple Output Test:\n");
    for (int i = 0; i < OUTPUTS; i++) {
        printf("Output %d: %f\n", i, results[i]);
    }
}

// Test for Multiple Input Multiple Output neural network
void test_mimo_nn() {
    double inputs[NUM_FEATURES] = {30, 87, 100};
    double weights[OUTPUTS][NUM_FEATURES] = {
        {-2, 9.5, 2.01},
        {-0.8, 7.2, 6.3},
        {-0.5, 0.45, 0.9}
    };
    double results[OUTPUTS];

    multiple_in_multiple_out(inputs, NUM_FEATURES, results, OUTPUTS, weights);

    printf("Multiple Input Multiple Output Test:\n");
    for (int i = 0; i < OUTPUTS; i++) {
        printf("Output %d: %f\n", i, results[i]);
    }
}

// Test for Hidden Layer neural network
void test_hidden_layer_nn() {
    Layer hidden_layer = create_layer(INPUT_SIZE, HIDDEN_SIZE, SIGMOID);
    Layer output_layer = create_layer(HIDDEN_SIZE, OUTPUT_SIZE, SIGMOID);
    double inputs[INPUT_SIZE] = {30, 82, 110};
    double output_vector[OUTPUT_SIZE];

    // Initialize weights for hidden and output layers
    random_weight_initialization(HIDDEN_SIZE, INPUT_SIZE, hidden_layer.weights);
    random_weight_initialization(OUTPUT_SIZE, HIDDEN_SIZE, output_layer.weights);

    // Forward pass through the hidden layer and output layer
    hidden_layer_nn(inputs, &hidden_layer, &output_layer, output_vector);

    printf("Hidden Layer Neural Network Test:\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
    printf("Predicted Output %d: %f\n", i, output_vector[i]);
    }

    // Example expected values
    double expected_values[OUTPUT_SIZE] = {30, 87, 110};
    printf("Errors:\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("Error for output %d: %f\n", i, find_error_simple(output_vector[i], expected_values[i]));
    }
}

// Test for brute force learning
void test_brute_force() {
    double input = 0.5;
    double weight = 0.5;
    double expected_value = 0.8;
    double step_amount = 0.001;
    uint32_t iterations = 800;

    printf("Brute Force Learning Test:\n");
}

// Forward propagation example with a basic neural network
void forward_propagation() {
    double raw_x_data[NUM_FEATURES][NUM_EXAMPLES] = {
        {2, 5, 1},
        {8, 5, 8}
    };
    double raw_y_data[1][NUM_EXAMPLES] = {{200, 90, 190}};

    double **raw_x = (double **)malloc(NUM_FEATURES * sizeof(double *));
    double **raw_y = (double **)malloc(1 * sizeof(double *));
    double **train_x = (double **)malloc(NUM_FEATURES * sizeof(double *));
    double **train_y = (double **)malloc(1 * sizeof(double *));

    for (int i = 0; i < NUM_FEATURES; i++) {
        raw_x[i] = raw_x_data[i];
        train_x[i] = (double *)malloc(NUM_EXAMPLES * sizeof(double));
    }
    raw_y[0] = raw_y_data[0];
    train_y[0] = (double *)malloc(NUM_EXAMPLES * sizeof(double));

    normalize_data_2D(NUM_FEATURES, NUM_EXAMPLES, raw_x, train_x);
    normalize_data_2D(1, NUM_EXAMPLES, raw_y, train_y);

    double syn0[NUM_OF_HID_NODES][NUM_FEATURES] = {
        {-0.1, 0.2, 0.3},
        {0.4, -0.5, 0.6},
        {-0.7, 0.8, -0.9}
    };
    double syn1[NUM_OF_OUT_NODES][NUM_OF_HID_NODES] = {
        {0.1, -0.2, 0.3}
    };

    double train_x_eg1[NUM_FEATURES];
    double train_y_eg1 = train_y[0][0];

    for (int i = 0; i < NUM_FEATURES; i++) {
        train_x_eg1[i] = train_x[i][0];
    }

    double z1_eg1[NUM_OF_HID_NODES];
    double a1_eg1[NUM_OF_HID_NODES];
    double z2_eg1 = 0;
    double yhat_eg1 = 0;

    // Forward pass through the hidden layer
    for (int i = 0; i < NUM_OF_HID_NODES; i++) {
        z1_eg1[i] = weighted_sum(train_x_eg1, syn0[i], NUM_FEATURES);
        a1_eg1[i] = 1 / (1 + exp(-z1_eg1[i]));  // Sigmoid activation
    }

    // Forward pass through the output layer
    z2_eg1 = weighted_sum(a1_eg1, syn1[0], NUM_OF_HID_NODES);
    yhat_eg1 = 1 / (1 + exp(-z2_eg1));  // Sigmoid activation

    printf("Normalized training example: [%f, %f]\n", train_x_eg1[0], train_x_eg1[1]);
    printf("Normalized training label: %f\n", train_y_eg1);
    printf("Predicted output: %f\n", yhat_eg1);

    // Cleanup
    for (int i = 0; i < NUM_FEATURES; i++) {
        free(train_x[i]);
    }
    free(raw_x);
    free(raw_y);
    free(train_x);
    free(train_y);
}*/
