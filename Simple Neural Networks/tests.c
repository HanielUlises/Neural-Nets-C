#include "tests.h"

void test_siso_nn (){
    double temperature[] = {12, 23, 50, -10, 16};
    double weight = -2;

    size_t vec_size = sizeof(temperature) / sizeof(double);

    printf("Predicted values: \n");
    for(size_t i = 0; i < vec_size; i++){
        printf("Current predicted value %zu: %f\n", i+1, single_in_single_out(temperature[i], weight));
    }
}

void test_miso_nn() {
    double temperature[] = {12, 23, 50, -10, 16};
    double humidity[] = {60, 67, 50, 65, 63};
    double air_quality[] = {60, 47, 167, 187, 94};

    size_t vec_size = sizeof(temperature) / sizeof(double);

    double weights[] = {-2, 2, 1};

    double training_eg[3]; // Three features: temperature, humidity, air quality

    for (size_t i = 0; i < vec_size; i++) {
        training_eg[0] = temperature[i];
        training_eg[1] = humidity[i];
        training_eg[2] = air_quality[i];

        printf("Current test: %f \r\n", multiple_in_single_out(training_eg, weights, 3));
    }
}

void test_simo_nn(){
    const double Sad = 0.9;
    // These weights are for: 
    // Temperature
    // Humidity
    // Air quality
    double weights [3] = {-20, 95, 201};
    double predicted_results[3];

    size_t size = sizeof(weights)/sizeof(double);

    single_in_multiple_out(Sad, weights, predicted_results, size);

    printf("Predicted temperature is: %f \r\n", predicted_results[0]);
    printf("Predicted humidity is: %f \r\n", predicted_results[1]);
    printf("Predicted air quality is: %f \r\n", predicted_results[2]);
}

void test_mimo_nn (){
    double predicted_results[3];

    size_t size = sizeof(predicted_results)/sizeof(double);

    double weights[NUM_FEATURES][OUTPUTS] = 
        {{-2, 9.5, 2.01},
         {-0.8, 7.2, 6.3},
         {-0.5, 0.45, 0.9}};

    double inputs[NUM_FEATURES] = {30, 87, 100};

    multiple_in_multiple_out(inputs, NUM_FEATURES,predicted_results, OUTPUTS, weights);

    printf("Predicted values of multiple input, multiple output NN\n");
    for(size_t i = 0; i < size; i++){
        printf("%d %f\n",i+1, predicted_results[i]);
    }
}

void test_hidden_layer_nn (){
    double predicted_results[3];
    size_t predict_size = sizeof(predicted_results)/sizeof(double);
    // Vector from input to the actual hidden layer
    double input_to_hidden[HIDDEN_SIZE][INPUT_SIZE] = 
                                                        {{-2, 9.5, 2.01},
                                                         {-0.8,7.2, 6.3},
                                                         {-0.5, 0.45, 0.9}};
    // Vector from hidden layer to the output vector
    double hidden_to_output[OUTPUT_SIZE][HIDDEN_SIZE] =
                                                        {{-1.0, 1.15, 0.11},
                                                         {-0.18, 0.15, -0.01},
                                                         {0.25, -0.25, -0.1}};
    double inputs[INPUT_SIZE] = {30, 82, 110};
    hidden_layer_nn(inputs, input_to_hidden, hidden_to_output, predicted_results);

    printf("Predicted results:   \n");
    for(size_t i = 0; i < predict_size; i++){
        printf("%f", predicted_results[i]);
    }

    double expected_values [OUTPUT_SIZE] = {30, 87, 110};
    printf("Sad error: %f", find_error_simple(predicted_results[SAD_IDX], expected_values[SAD_IDX]));
    printf("Sick error: %f", find_error_simple(predicted_results[SICK_IDX], expected_values[SICK_IDX]));
    printf("Active error: %f", find_error_simple(predicted_results[ACTIVE_IDX], expected_values[ACTIVE_IDX]));
}

void test_brute_force(){
    double input = 0.5;
    double weight = 0.5;
    double expected_value = 0.8;
    double step_amount = 0.001;

    bruteforce_learning(input, weight, expected_value, step_amount, 800);
}

void forward_propagation() {
    double raw_x_data[NUM_FEATURES][NUM_EXAMPLES] = {
        {2, 5, 1},
        {8, 5, 8}
    };

    double raw_y_data[1][NUM_EXAMPLES] = {{200, 90, 190}};

    double **raw_x = malloc(NUM_FEATURES * sizeof(double *));
    double **raw_y = malloc(1 * sizeof(double *));
    double **train_x = malloc(NUM_FEATURES * sizeof(double *));
    double **train_y = malloc(1 * sizeof(double *));

    for (int i = 0; i < NUM_FEATURES; i++) {
        raw_x[i] = raw_x_data[i];
        train_x[i] = malloc(NUM_EXAMPLES * sizeof(double));
    }
    raw_y[0] = raw_y_data[0];
    train_y[0] = malloc(NUM_EXAMPLES * sizeof(double));

    // Normalize data
    normalize_data_2D(NUM_FEATURES, NUM_EXAMPLES, raw_x, train_x);
    normalize_data_2D(1, NUM_EXAMPLES, raw_y, train_y);

    // Neural network weights
    double syn0[NUM_OF_HID_NODES][NUM_FEATURES] = {
        {-0.1, 0.2, 0.3},
        {0.4, -0.5, 0.6},
        {-0.7, 0.8, -0.9}
    };
    double syn1[NUM_OF_OUT_NODES][NUM_OF_HID_NODES] = {
        {0.1, -0.2, 0.3}
    };

    double train_x_eg1[NUM_FEATURES];
    double train_y_eg1;

    train_x_eg1[0] = train_x[0][0];
    train_x_eg1[1] = train_x[1][0];

    train_y_eg1 = train_y[0][0];

    // Neural network operations
    double z1_eg1[NUM_OF_HID_NODES];
    double a1_eg1[NUM_OF_HID_NODES];
    double z2_eg1;
    double yhat_eg1;

    // Forward propagation through the network
    // Layer 1 (Input to Hidden Layer)
    for (int i = 0; i < NUM_OF_HID_NODES; i++) {
        z1_eg1[i] = 0;
        for (int j = 0; j < NUM_FEATURES; j++) {
            z1_eg1[i] += train_x_eg1[j] * syn0[i][j];
        }
        // Applying activation function
        a1_eg1[i] = 1 / (1 + exp(-z1_eg1[i]));
    }

    // Layer 2 (Hidden Layer to Output Layer)
    z2_eg1 = 0;
    for (int i = 0; i < NUM_OF_HID_NODES; i++) {
        z2_eg1 += a1_eg1[i] * syn1[0][i];
    }

    // Activation function (sigmoid)
    yhat_eg1 = 1 / (1 + exp(-z2_eg1));

    printf("Normalized training example: [%f, %f]\n", train_x_eg1[0], train_x_eg1[1]);
    printf("Normalized training label: %f\n", train_y_eg1);
    printf("Predicted output: %f\n", yhat_eg1);

    for (int i = 0; i < NUM_FEATURES; i++) {
        free(train_x[i]);
    }
    free(raw_x);
    free(raw_y);
    free(train_x);
    free(train_y);
}