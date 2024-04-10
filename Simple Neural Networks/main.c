#include <stdio.h>
#include <stdlib.h>

#include "simple_nn.h"

void test_nn(){
    double temperature[] = {12, 23, 50, -10, 16};
    double weight = -2;

    size_t vec_size = sizeof(temperature) / sizeof(double);

    printf("Predicted values: \n");
    for(size_t i = 0; i < vec_size; i++){
        printf("Current predicted value %zu: %f\n", i, single_in_single_out(temperature[i], weight));
    }
}


int main (){
    test_nn();
    return 0;
}