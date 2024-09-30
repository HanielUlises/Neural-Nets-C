#include "simple_nn.h"

void gsd(double inputs[][2], double outputs[], int num_samples) {
    // Ensure that num_samples is an integral value
    if (num_samples <= 0) {
        fprintf(stderr, "Number of samples must be greater than zero.\n");
        return;
    }
    
    // XOR-like samples
    for (int i = 0; i < num_samples; i++) {
        // Alternating 0 and 1
        inputs[i][0] = (double)(i % 2); 
        inputs[i][1] = (double)((i / 2) % 2); 
        // XOR output
        outputs[i] = (inputs[i][0] != inputs[i][1]) ? 1.0 : 0.0; 
    }
}

int main() {
    int layer_sizes[] = {2, 3, 1};  // Input layer with 2 neurons, 1 hidden layer with 3 neurons, and output layer with 1 neuron
    Activation activations[] = {SIGMOID, SIGMOID, SIGMOID};  

    // Neural network with 2 hidden layers
    NeuralNetwork nn = create_neural_network(3, layer_sizes, activations);

    // Weight init
    for (int i = 0; i < 3; i++) {
        random_weight_initialization(layer_sizes[i], layer_sizes[i + 1], nn.layers[i].weights);
    }

    // Synthetic data for training (4 samples for XOR)
    double inputs[4][2];
    double outputs[4];
    gsd(inputs, outputs, 4);

    // Hyperparameters
    double learning_rate = 0.1;
    uint32_t iterations = 10000;
    // L2 regularization with lambda = 0.01
    Regularizer regularizer = {L2, 0.01}; 

    // Training loop
    for (int iter = 0; iter < iterations; iter++) {
        for (int sample = 0; sample < 4; sample++) {
            // Forward pass
            forward_pass(&nn, inputs[sample]);

            double loss = compute_loss(MEAN_SQUARED_ERROR, nn.output_vector, &outputs[sample], 1);
            backpropagation(&nn, inputs[sample], &outputs[sample], learning_rate);

            if (iter % 1000 == 0) {
                printf("Iteration: %d, Sample: %d, Loss: %f\n", iter, sample, loss);
            }
        }
    }

    printf("Testing the trained model:\n");
    for (int sample = 0; sample < 4; sample++) {
        forward_pass(&nn, inputs[sample]);
        double predicted_value = nn.output_vector[0] >= 0.5 ? 1.0 : 0.0;
        printf("Input: [%f, %f], Predicted: %f, Expected: %f\n", inputs[sample][0], inputs[sample][1], predicted_value, outputs[sample]);
    }

    destroy_neural_network(&nn);

    return 0;
}
