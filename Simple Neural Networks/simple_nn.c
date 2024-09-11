#include "simple_nn.h"

// Multiply a single input by a weight to produce a single output.
double single_in_single_out(double input, double weight) {
    return input * weight;
}

// Weighted sum of multiple inputs against corresponding weights.
double multiple_in_single_out(double* input, double* weight, int length) {
    return weighted_sum(input, weight, length);
}

void single_in_multiple_out(double scalar, double* w_vect, double* out_vect, int length) {
    element_wise_multiply(scalar, w_vect, out_vect, length);
}

// Computes the output vector from an input vector and a matrix of weights.
void multiple_in_multiple_out(double *input_vector, int INPUT_LEN, double *output_vector, int OUTPUT_LEN, double **weight_matrix) {
    for (int i = 0; i < OUTPUT_LEN; i++) {
        output_vector[i] = 0;
        for (int j = 0; j < INPUT_LEN; j++) {
            output_vector[i] += input_vector[j] * weight_matrix[j][i];
        }
    }
}

// Create a new layer with random weights and biases
Layer create_layer(int input_size, int output_size, Activation activation) {
    Layer layer;
    layer.input_size = input_size;
    layer.output_size = output_size;
    layer.activation = activation;

    // Allocate memory for weights and biases
    layer.weights = (double **)malloc(output_size * sizeof(double *));
    for (int i = 0; i < output_size; i++) {
        layer.weights[i] = (double *)malloc(input_size * sizeof(double));
    }
    layer.biases = (double *)malloc(output_size * sizeof(double));

    // Random weight and bias initialization
    random_weight_initialization(output_size, input_size, layer.weights);
    random_weight_init_1D(layer.biases, output_size);

    return layer;
}

// Create a new neural network with specified layers and activation functions
NeuralNetwork create_neural_network(int num_layers, int *layer_sizes, Activation *activations) {
    NeuralNetwork nn;
    nn.num_layers = num_layers;
    nn.layers = (Layer *)malloc(num_layers * sizeof(Layer));

    for (int i = 0; i < num_layers; i++) {
        int input_size = (i == 0) ? layer_sizes[i] : layer_sizes[i - 1];
        int output_size = layer_sizes[i];
        nn.layers[i] = create_layer(input_size, output_size, activations[i]);
    }

    nn.output_vector = (double *)malloc(layer_sizes[num_layers - 1] * sizeof(double));

    return nn;
}

// Forward pass through the neural network
void forward_pass(NeuralNetwork *nn, double *input_vector) {
    double *current_input = input_vector;
    double *current_output = NULL;

    for (int layer_idx = 0; layer_idx < nn->num_layers; layer_idx++) {
        Layer *layer = &nn->layers[layer_idx];
        current_output = (double *)malloc(layer->output_size * sizeof(double));

        matrix_vector_multiplication(current_input, layer->input_size, current_output, layer->output_size, layer->weights);

        for (int i = 0; i < layer->output_size; i++) {
            current_output[i] += layer->biases[i];

            switch (layer->activation) {
                case RELU:
                    current_output[i] = fmax(0.0, current_output[i]);
                    break;
                case SIGMOID:
                    current_output[i] = 1.0 / (1.0 + exp(-current_output[i]));
                    break;
                case SOFTMAX:
                    // Softmax will be applied at the output layer
                    break;
                case NONE:
                    // No activation
                    break;
            }
        }

        if (layer_idx == nn->num_layers - 1) {
            for (int i = 0; i < layer->output_size; i++) {
                nn->output_vector[i] = current_output[i];
            }
        } else {
            current_input = current_output;
        }

        free(current_output);
    }

    if (nn->layers[nn->num_layers - 1].activation == SOFTMAX) {
        softmax(nn->output_vector, nn->output_vector, nn->layers[nn->num_layers - 1].output_size);
    }
}

// Matrix-vector multiplication
void matrix_vector_multiplication(double *input_vector, int INPUT_LEN, double *output_vector, int OUTPUT_LEN, double **weight_matrix) {
    for (int i = 0; i < OUTPUT_LEN; i++) {
        output_vector[i] = 0;
        for (int j = 0; j < INPUT_LEN; j++) {
            output_vector[i] += input_vector[j] * weight_matrix[i][j];
        }
    }
}

// Weighted sum of an array of inputs with an array of weights.
double weighted_sum(double* input, double* weight, int length) {
    double output = 0.0;
    for (int i = 0; i < length; i++) {
        output += input[i] * weight[i];
    }
    return output;
}

// Element-wise multiplication of a scalar with each element in a vector.
void element_wise_multiply(double input_scalar, double* weight_vector, double* output_vector, int length) {
    for (int i = 0; i < length; i++) {
        output_vector[i] = input_scalar * weight_vector[i];
    }
}

// Calculates squared error of a prediction based on input and expected value.
double find_error(double input, double weight, double expected_value) {
    return pow(((input * weight) - expected_value), 2);
}

// Simple squared error between prediction and actual value.
double find_error_simple(double yhat, double y) {
    return pow((yhat - y), 2);
}

// Perform brute force learning to find a better weight
void bruteforce_learning(double input, double weight, double expected_values, double step_amount, uint32_t itr) {
    double prediction, error;
    double up_prediction, up_error;
    double down_prediction, down_error;

    for (int i = 0; i < itr; i++) {
        prediction = input * weight;
        error = pow((prediction - expected_values), 2);
        printf("Error: %f Prediction: %f \r\n", error, prediction);

        up_prediction = input * (weight + step_amount);
        up_error = pow((expected_values - up_prediction), 2);

        down_prediction = input * (weight - step_amount);
        down_error = pow((expected_values - down_prediction), 2);

        if (down_error < up_error) {
            weight -= step_amount;
        } else if (down_error > up_error) {
            weight += step_amount;
        }
    }
}

// Normalize data by dividing each element by the max value
void normalize_data(double *input_vector, double *output_vector, int LEN) {
    double max = input_vector[0];

    for (int i = 0; i < LEN; i++) {
        if (input_vector[i] > max) {
            max = input_vector[i];
        }
    }

    for (int i = 0; i < LEN; i++) {
        output_vector[i] = input_vector[i] / max;
    }
}

// Normalize a 2D matrix
void normalize_data_2D(int row, int col, double **input_matrix, double **output) {
    double max = -DBL_MAX;

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (input_matrix[i][j] > max) {
                max = input_matrix[i][j];
            }
        }
    }

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            output[i][j] = input_matrix[i][j] / max;
        }
    }
}

// Randomly initialize 2D array of weights
void random_weight_initialization(int rows, int cols, double **weights_matrix) {
    srand(2);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            weights_matrix[i][j] = ((double)rand() / RAND_MAX);
        }
    }
}

// Randomly initialize 1D array of weights
void random_weight_init_1D(double *output_vector, uint32_t LEN) {
    srand(2);
    for (int i = 0; i < LEN; i++) {
        output_vector[i] = ((double)rand() / RAND_MAX);
    }
}

// Softmax function for normalizing outputs to probabilities
void softmax(double *input_vector, double *output_vector, int length) {
    double max = input_vector[0];
    double sum = 0.0;

    for (int i = 1; i < length; i++) {
        if (input_vector[i] > max) {
            max = input_vector[i];
        }
    }

    for (int i = 0; i < length; i++) {
        output_vector[i] = exp(input_vector[i] - max);
        sum += output_vector[i];
    }

    for (int i = 0; i < length; i++) {
        output_vector[i] /= sum;
    }
}

// ReLU activation function
void relu(double *input_vector, double *output_vector, int length) {
    for (int i = 0; i < length; i++) {
        output_vector[i] = fmax(0.0, input_vector[i]);
    }
}

// Sigmoid activation function
void sigmoid(double *input_vector, double *output_vector, int length) {
    for (int i = 0; i < length; i++) {
        output_vector[i] = 1.0 / (1.0 + exp(-input_vector[i]));
    }
}

/**
 * Performs a forward pass through a deep neural network with multiple layers.
 * This function computes the output of the network given an input vector, a list of layers, and their sizes.
 * 
 * @param input_vector A pointer to the input data vector.
 * @param input_size The size of the input vector (number of features).
 * @param output_vector A pointer to the vector where the final output of the network will be stored.
 * @param output_size The size of the output vector (number of output neurons).
 * @param layers A pointer to an array of Layer structs representing each layer of the network.
 * @param num_layers The total number of layers in the network.
 */
void deep_nn(double *input_vector, int input_size,
             double *output_vector, int output_size, 
             Layer *layers, int num_layers) {

    double *current_input = input_vector;  // Input to the first layer is the input_vector
    double *current_output = NULL;         // Temporary storage for each layer's output

    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        Layer *current_layer = &layers[layer_idx];
        current_output = (double *)malloc(current_layer->output_size * sizeof(double));

        // Perform matrix-vector multiplication
        matrix_vector_multiplication(current_input, current_layer->input_size, current_output, current_layer->output_size, current_layer->weights);

        // Add biases and apply activation function
        for (int i = 0; i < current_layer->output_size; i++) {
            current_output[i] += current_layer->biases[i]; // Adding bias

            // Apply activation function
            switch (current_layer->activation) {
                case RELU:
                    current_output[i] = fmax(0.0, current_output[i]);  // ReLU activation
                    break;
                case SIGMOID:
                    current_output[i] = 1.0 / (1.0 + exp(-current_output[i])); // Sigmoid activation
                    break;
                case SOFTMAX:
                    // Softmax will be applied at the final output layer
                    break;
                case NONE:
                    // No activation for linear layers
                    break;
            }
        }

        // If it's the last layer, copy the results to output_vector
        if (layer_idx == num_layers - 1) {
            for (int i = 0; i < output_size; i++) {
                output_vector[i] = current_output[i];
            }
        } else {
            // For intermediate layers, pass the current output as the next layer's input
            current_input = current_output;
        }

        free(current_output);  // Free memory allocated for current layer's output
    }

    // If the final layer uses softmax, apply it to the final output
    if (layers[num_layers - 1].activation == SOFTMAX) {
        softmax(output_vector, output_vector, output_size);
    }
}
