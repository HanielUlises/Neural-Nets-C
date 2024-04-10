#include <stdio.h>

#include "simple_nn.h"

void test_siso_nn (){
    double temperature[] = {12, 23, 50, -10, 16};
    double weight = -2;

    size_t vec_size = sizeof(temperature) / sizeof(double);

    printf("Predicted values: \n");
    for(size_t i = 0; i < vec_size; i++){
        printf("Current predicted value %zu: %f\n", i, single_in_single_out(temperature[i], weight));
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

    int size = sizeof(weights)/sizeof(double);

    single_in_multiple_out(Sad, weights, predicted_results, size);

    printf("Predicted temperature is: %f \r\n", predicted_results[0]);
    printf("Predicted humidity is: %f \r\n", predicted_results[1]);
    printf("Predicted air quality is: %f \r\n", predicted_results[2]);
}

int main (){
    printf("====================================================\r\n");
    test_siso_nn();
    printf("====================================================\r\n");
    test_simo_nn();
    return 0;
}