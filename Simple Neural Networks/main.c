#include "simple_nn.h"

int main() {
    int layer_sizes[] = {3, 3, 3}; 
    Activation activations[] = {RELU, RELU, SOFTMAX};
    NeuralNetwork nn = create_neural_network(3, layer_sizes, activations);

    double input_vector[] = {0.5, 0.8, 0.3};  
    double output_vector[3];

    forward_pass(&nn, input_vector);

    printf("Output after forward pass:\n");
    for (int i = 0; i < 3; i++) {
        printf("%f ", nn.output_vector[i]);
    }
    printf("\n");

    // Cleaning up
    for (int i = 0; i < nn.num_layers; i++) {
        for (int j = 0; j < nn.layers[i].output_size; j++) {
            free(nn.layers[i].weights[j]);
        }
        free(nn.layers[i].weights);
        free(nn.layers[i].biases);
    }
    free(nn.layers);
    free(nn.output_vector);

    return 0;
}
