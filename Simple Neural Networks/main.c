#include "simple_nn.h"

int main() {
    int layer_sizes[] = {2, 1};  // One layer with 2 inputs and 1 output
    Activation activations[] = {SIGMOID};  

    // Create a neural network with 1 layer
    NeuralNetwork nn = create_neural_network(1, layer_sizes, activations);

    // Set weights and bias
    nn.layers[0].weights[0][0] = 0.6;  
    nn.layers[0].weights[0][1] = 0.8;  
    nn.layers[0].biases[0] = 0;        

    // Input vectors
    double input_vector_1[] = {1.0, 6.0};
    double input_vector_2[] = {2.0, 8.0};
    
    double expected_output = 0.0; 

    // Testing neuron 1... Calculating the vector product
    printf("==============================================\n");
    printf("-Neuron 1\n");
    // Evaluating the product with the sigmoid function
    forward_pass(&nn, input_vector_1);
    printf("==============================================\n");
    printf("-Output after forward pass (sigmoid output): %f\n", nn.output_vector[0]);

    double prediction_1 = nn.output_vector[0];
    double loss_1 = - (expected_output * log(prediction_1) + (1 - expected_output) * log(1 - prediction_1));
    printf("-Loss for input_vector_1: %f\n", loss_1);

    double predicted_value_1 = prediction_1 >= 0.5 ? 1.0 : 0.0;
    printf("-Predicted value for input_vector_1: %f\n", predicted_value_1);

    printf("==============================================\n");
    // Testing neuron 2...Calculating the vector product
    printf("-Neuron 2\n");
    forward_pass(&nn, input_vector_2);
    // Evaluating the product with the sigmoid function
    printf("==============================================\n");
    printf("-Output after forward pass (sigmoid output): %f\n", nn.output_vector[0]);

    double prediction_2 = nn.output_vector[0];
    double loss_2 = - (expected_output * log(prediction_2) + (1 - expected_output) * log(1 - prediction_2));
    printf("-Loss for input_vector_2: %f\n", loss_2);

    double predicted_value_2 = prediction_2 >= 0.5 ? 1.0 : 0.0;
    printf("-Predicted value for input_vector_2: %f\n", predicted_value_2);

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
