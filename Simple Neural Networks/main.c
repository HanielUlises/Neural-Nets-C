#include <stdio.h>

#include "simple_nn.h"

#define SAD_IDX     0
#define SICK_IDX    1
#define ACTIVE_IDX  2

/*
    Problem is to determine whether a person is sad or not based on 
    the ambience

    Prediction: Person's mood.
*/

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

int main (){
    printf("====================================================\r\n");
    test_siso_nn();
    printf("====================================================\r\n");
    test_simo_nn();
    printf("====================================================\r\n");
    test_miso_nn();
    printf("====================================================\r\n");
    test_mimo_nn();
    printf("====================================================\r\n");
    printf("Hidden layer neural network\n");
    test_hidden_layer_nn();
    return 0;
}