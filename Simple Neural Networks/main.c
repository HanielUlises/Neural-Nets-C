#include "simple_nn.h"

int main() {
    // Two entry neurons with a single output neuron
    int layer_sizes[] = {2, 1};  

    Activation activations[] = {RELU};  
    NeuralNetwork nn = create_neural_network(1, layer_sizes, activations);

    nn.layers[0].weights[0][0] = 0.6;
    nn.layers[0].weights[0][1] = 0.8;  
    nn.layers[0].biases[0] = 0;       

    double input_vector1[] = {1, 6};
    forward_pass(&nn, input_vector1);

    printf("Output for input [1, 6]: %f\n", nn.output_vector[0]);

    double input_vector2[] = {2, 8};
    forward_pass(&nn, input_vector2);

    printf("Output for input [2, 8]: %f\n", nn.output_vector[0]);

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
