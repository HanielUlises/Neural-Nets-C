#include <stdio.h>
#include <stdlib.h>

#include "simple_nn.h"

void test_nn(){
    double temperature[] = {12,23,50,-10,16};
    double weight = -2;

    printf("The first predicted value is %d: \r\n", single_in_single_out(temperature[0],weight));
}

int main (){
    test_nn();
    return 0;
}