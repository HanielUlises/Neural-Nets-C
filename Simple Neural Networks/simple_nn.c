#include "simple_nn.h"

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

Optimizer create_optimizer(OptimizerType type, double learning_rate, double momentum, double beta1, double beta2, double epsilon) {
    Optimizer opt;
    opt.type = type;
    opt.learning_rate = learning_rate;
    opt.momentum = momentum;
    opt.beta1 = beta1;
    opt.beta2 = beta2;
    opt.epsilon = epsilon;

    return opt;
}

// Memory cleanup (destructors)
void destroy_layer(Layer *layer) {
    for (int i = 0; i < layer->output_size; i++) {
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer->biases);
}

void destroy_neural_network(NeuralNetwork *nn) {
    for (int i = 0; i < nn->num_layers; i++) {
        destroy_layer(&nn->layers[i]);
    }
    free(nn->layers);
    free(nn->output_vector);
    free(momentum_velocity);
    free(m_t);
    free(v_t);
}

static void apply_activation(double *output_vector, int size, Activation activation) {
    for (int i = 0; i < size; i++) {
        switch (activation) {
            case RELU:
                output_vector[i] = fmax(0.0, output_vector[i]);
                break;
            case SIGMOID:
                output_vector[i] = 1.0 / (1.0 + exp(-output_vector[i]));
                break;
            case SOFTMAX:
                break;
            case NO_ACTIVATION:
                break;
        }
    }
}

// Normalize data by dividing each element base on the max value
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

// Activation functions and loss functions
// i) Activation functions (not explicitly used right now)
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

void softmax_derivative(double *input_vector, double *output_vector, int length) {
    // placeholder    
}

// ReLU activation function
void relu(double *input_vector, double *output_vector, int length) {
    for (int i = 0; i < length; i++) {
        output_vector[i] = fmax(0.0, input_vector[i]);
    }
}

void relu_derivative(double *input_vector, double *output_vector, int length){
    for (int i = 0; i < length; i++) {
        output_vector[i] = (input_vector[i] > 0) ? 1.0 : 0.0;
    }
}

// Sigmoid activation function
void sigmoid(double *input_vector, double *output_vector, int length) {
    for (int i = 0; i < length; i++) {
        output_vector[i] = 1.0 / (1.0 + exp(-input_vector[i]));
    }
}

// Softmax function derivative for optimization
void sigmoid_derivative(double *input_vector, double *output_vector, int length) {
    for (int i = 0; i < length; i++) {
        output_vector[i] = input_vector[i] * (1.0 - input_vector[i]);
    }
}

static void apply_derivative(double *output_vector, int size, Derivative derivative){
    switch (derivative) {
        case RELU_P:
            relu_derivative(output_vector, output_vector, size);
            break;
        case SIGMOID_P:
            sigmoid_derivative(output_vector, output_vector, size);
            break;
        case SOFTMAX_P:
            softmax_derivative(output_vector, output_vector, size);
            break;
        case NO_DERIVATIVE:
            // Do nothing
            break;
    }
}

// ii) Loss functions
// Mean Squared Error (MSE) Loss Function
double mean_squared_error(double *predicted, double *actual, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        double error = predicted[i] - actual[i];
        sum += error * error;
    }
    return sum / (double)size;
}

// Derivative of Mean Squared Error (MSE) for backpropagation
void mean_squared_error_derivative(double *predicted, double *actual, double *derivative_out, int size) {
    for (int i = 0; i < size; i++) {
        derivative_out[i] = 2.0 * (predicted[i] - actual[i]) / size;
    }
}

// Cross-Entropy Loss Function (for classification tasks)
// (used with softmax)
double cross_entropy_loss(double *predicted, double *actual, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        // As log(0) is an error, we will use a 
        // infinitesimal number
        double epsilon = 1e-12;
        sum += -actual[i] * log(predicted[i] + epsilon);
    }
    return sum / (double)size;
}

// Derivative of Cross-Entropy Loss for backpropagation 
// (used with softmax)
void cross_entropy_loss_derivative(double *predicted, double *actual, double *derivative_out, int size) {
    for (int i = 0; i < size; i++) {
        derivative_out[i] = predicted[i] - actual[i];
    }
}

double compute_loss(LossFunction loss_function, double *predicted, double *actual, int size) {
    if (loss_function == MEAN_SQUARED_ERROR) {
        return mean_squared_error(predicted, actual, size);
    } else if (loss_function == CROSS_ENTROPY) {
        return cross_entropy_loss(predicted, actual, size);
    }
    return 0.0; 
}

void compute_loss_derivative(LossFunction loss_function, double *predicted, double *actual, double *derivative_out, int size) {
    if (loss_function == MEAN_SQUARED_ERROR) {
        mean_squared_error_derivative(predicted, actual, derivative_out, size);
    } else if (loss_function == CROSS_ENTROPY) {
        cross_entropy_loss_derivative(predicted, actual, derivative_out, size);
    }
}

// Optimization
void update_weights(Optimizer *optimizer, double *weights, double *gradients, int length, Regularizer *regularizer) {
    if (optimizer->type == SGD) {
        // Stochastic Gradient Descent (SGD) update
        for (int i = 0; i < length; i++) {
            weights[i] -= optimizer->learning_rate * gradients[i];

            // Apply L1 regularization
            if (regularizer->reg_type == L1) {
                weights[i] -= regularizer->lambda * (weights[i] > 0 ? 1 : -1);
            }
            // Apply L2 regularization
            else if (regularizer->reg_type == L2) {
                weights[i] -= regularizer->lambda * weights[i];
            }
        }
    } else if (optimizer->type == MOMENTUM) {
        if (momentum_velocity == NULL) {
            momentum_velocity = (double *)malloc(length * sizeof(double));
            memset(momentum_velocity, 0, length * sizeof(double));
        }

        for (int i = 0; i < length; i++) {
            // Update velocity: v = momentum * v - learning_rate * gradient
            momentum_velocity[i] = optimizer->momentum * momentum_velocity[i] - optimizer->learning_rate * gradients[i];

            // Update weight: w = w + v
            weights[i] += momentum_velocity[i];

            // Apply L1 regularization
            if (regularizer->reg_type == L1) {
                weights[i] -= regularizer->lambda * (weights[i] > 0 ? 1 : -1);
            }
            // Apply L2 regularization
            else if (regularizer->reg_type == L2) {
                weights[i] -= regularizer->lambda * weights[i];
            }
        }
    } else if (optimizer->type == ADAM) {
        if (m_t == NULL || v_t == NULL) {
            m_t = (double *)malloc(length * sizeof(double));
            v_t = (double *)malloc(length * sizeof(double));
            memset(m_t, 0, length * sizeof(double)); // First moment estimate to zero
            memset(v_t, 0, length * sizeof(double)); // Second moment estimate to zero
        }

        for (int i = 0; i < length; i++) {
            // Update biased first moment estimate: m_t = beta1 * m_t + (1 - beta1) * gradient
            m_t[i] = optimizer->beta1 * m_t[i] + (1 - optimizer->beta1) * gradients[i];

            // Update biased second moment estimate: v_t = beta2 * v_t + (1 - beta2) * (gradient^2)
            v_t[i] = optimizer->beta2 * v_t[i] + (1 - optimizer->beta2) * (gradients[i] * gradients[i]);

            // Compute bias-corrected first and second moment estimates
            double m_t_hat = m_t[i] / (1 - pow(optimizer->beta1, 2)); // Unbiased first moment
            double v_t_hat = v_t[i] / (1 - pow(optimizer->beta2, 2)); // Unbiased second moment

            // Update weights: w = w - learning_rate * m_t_hat / (sqrt(v_t_hat) + epsilon)
            weights[i] -= optimizer->learning_rate * m_t_hat / (sqrt(v_t_hat) + optimizer->epsilon);

            // Apply L1 regularization
            if (regularizer->reg_type == L1) {
                weights[i] -= regularizer->lambda * (weights[i] > 0 ? 1 : -1);
            }
            // Apply L2 regularization
            else if (regularizer->reg_type == L2) {
                weights[i] -= regularizer->lambda * weights[i];
            }
        }
    }
}

/**
 * Performs brute-force optimization of weights and biases for a specific layer in a neural network.
 * This function systematically explores slight perturbations of the weights and biases to minimize the 
 * difference between the predicted output and the expected values. It evaluates the impact of both 
 * increasing and decreasing each weight and bias by a specified learning rate.
 * 
 * This brute-force method is characterized by its exhaustive search approach, making it suitable 
 * for low-dimensional problems where the computational load is manageable. It is particularly useful 
 * in scenarios where gradient information is unreliable or unavailable, and it can aid in debugging 
 * or quick prototyping by revealing how specific weights and biases influence the model's output.
 *
 * @param input_vector A pointer to the input data vector.
 * @param expected_values A pointer to the expected output values for the given input.
 * @param learning_rate The step size used for perturbing weights and biases during optimization.
 * @param iterations The number of iterations to perform while searching for optimal parameters.
 * @param layer A pointer to the Layer struct representing the layer to optimize.
 */

/**
 * Performs a forward pass through a deep neural network with n layers.
 * This function computes the output of the network given an input vector, a list of layers, and their sizes.
 * 
 * @param nn Pointer to the NeuralNetwork structure that contains the layers, weights, biases, and activation functions.
 * @param input_vector Pointer to the input data vector, which represents the X input features to the network.
 * 
 * Process:
 * 1. Iterates through each layer of the neural network.
 * 2. For each layer:
 *    - Computes the weighted sum of inputs by performing matrix-vector multiplication.
 *    - Adds biases to the result.
 *    - Applies the specified activation function (e.g., RELU, SIGMOID, SOFTMAX).
 * 3. Passes the resulting output to the next layer as the input.
 * 4. Stores the final output of the network in the `nn->output_vector`.
 * 5. If the output layer uses the SOFTMAX activation function, it applies the Softmax function to the final output.
 * 
 * @note Ensure that the NeuralNetwork struct is properly initialized before invoking this function, i mean, its obvious.
 */
// Brute force learning to find a better weight for a specific layer
void bruteforce_learning(double *input_vector, double *expected_values, double learning_rate, uint32_t iterations, Layer *layer) {
    double *output_vector = (double *)malloc(layer->output_size * sizeof(double));
    if (!output_vector) {
        fprintf(stderr, "Memory allocation failed for output_vector\n");
        exit(EXIT_FAILURE);
    }

    double *best_weights = (double *)malloc(layer->input_size * sizeof(double));
    double *current_weights = (double *)malloc(layer->input_size * sizeof(double));
    double *best_biases = (double *)malloc(layer->output_size * sizeof(double));
    double *current_biases = (double *)malloc(layer->output_size * sizeof(double));
    if (!best_weights || !current_weights || !best_biases || !current_biases) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Initialize best weights and biases
    for (int i = 0; i < layer->output_size; i++) {
        best_biases[i] = layer->biases[i];
        for (int j = 0; j < layer->input_size; j++) {
            best_weights[j] = layer->weights[i][j];
        }
    }

    for (uint32_t iter = 0; iter < iterations; iter++) {
        // We're going to try perturbations in weights and biases
        double min_error = DBL_MAX;  // Start with a very high minimum error

        for (int i = 0; i < layer->output_size; i++) {
            for (int j = 0; j < layer->input_size; j++) {
                // Save current weight and bias
                current_weights[j] = layer->weights[i][j];
            }
            // Save current bias
            current_biases[i] = layer->biases[i];
        }

        // Testing perturbations for weights and biases
        for (int i = 0; i < layer->output_size; i++) {
            for (int j = 0; j < layer->input_size; j++) {
                // Testing weight increment
                layer->weights[i][j] = current_weights[j] + learning_rate;
                matrix_vector_multiplication(input_vector, layer->input_size, output_vector, layer->output_size, layer->weights);
                apply_activation(output_vector, layer->output_size, layer->activation);
                double error_up = compute_loss(layer->loss_func, output_vector, expected_values, layer->output_size);

                // Testing weight decrement
                layer->weights[i][j] = current_weights[j] - learning_rate;
                matrix_vector_multiplication(input_vector, layer->input_size, output_vector, layer->output_size, layer->weights);
                apply_activation(output_vector, layer->output_size, layer->activation);
                double error_down = compute_loss(layer->loss_func, output_vector, expected_values, layer->output_size);

                // Choose the best weight
                if (error_down < min_error) {
                    min_error = error_down;
                    best_weights[j] = current_weights[j] - learning_rate;
                } else if (error_up < min_error) {
                    min_error = error_up;
                    best_weights[j] = current_weights[j] + learning_rate;
                } else {
                    best_weights[j] = current_weights[j]; // Restore to current if no better found
                }

                // Restore the weight after testing
                layer->weights[i][j] = current_weights[j];
            }

            // Bias perturbation
            double best_bias = current_biases[i];
            layer->biases[i] = current_biases[i] + learning_rate;
            matrix_vector_multiplication(input_vector, layer->input_size, output_vector, layer->output_size, layer->weights);
            apply_activation(output_vector, layer->output_size, layer->activation);
            double error_up_bias = compute_loss(layer->loss_func, output_vector, expected_values, layer->output_size);

            layer->biases[i] = current_biases[i] - learning_rate;
            matrix_vector_multiplication(input_vector, layer->input_size, output_vector, layer->output_size, layer->weights);
            apply_activation(output_vector, layer->output_size, layer->activation);
            double error_down_bias = compute_loss(layer->loss_func, output_vector, expected_values, layer->output_size);

            // Choose the best bias
            if (error_down_bias < min_error) {
                min_error = error_down_bias;
                best_bias = current_biases[i] - learning_rate;
            } else if (error_up_bias < min_error) {
                min_error = error_up_bias;
                best_bias = current_biases[i] + learning_rate;
            }

            // Update the best biases found
            layer->biases[i] = best_bias;
        }

        // Apply the best weights found
        for (int i = 0; i < layer->output_size; i++) {
            for (int j = 0; j < layer->input_size; j++) {
                layer->weights[i][j] = best_weights[j];
            }
        }

        printf("Iteration %u: Minimum Error = %f\n", iter, min_error);
    }

    free(output_vector);
    free(best_weights);
    free(current_weights);
    free(best_biases);
    free(current_biases);
}

void forward_pass(NeuralNetwork *nn, double *input_vector) {
    double *current_input = input_vector;
    double *current_output = NULL;

    for (int layer_idx = 0; layer_idx < nn->num_layers; layer_idx++) {
        Layer *layer = &nn->layers[layer_idx];
        current_output = (double *)malloc(layer->output_size * sizeof(double));

        matrix_vector_multiplication(current_input, layer->input_size, current_output, layer->output_size, layer->weights);

        for (int i = 0; i < layer->output_size; i++) {
            current_output[i] += layer->biases[i];
            printf("Value %i output: %f\n", i, current_output[i]);
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
                case NO_ACTIVATION:
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

/**
 * Performs gradient descent optimization JUST for a single layer of a neural network.
 * This function updates the weights and biases of the layer by minimizing the loss 
 * between the predicted output and the expected values over a specified number of iterations.
 * 
 * @param input_vector A pointer to the input data vector for the current training instance.
 * @param expected_values A pointer to the expected output values (ground truth) for the input vector.
 * @param learning_rate The rate at which the weights and biases are updated during training.
 * @param iterations The number of iterations (or epochs) to perform the gradient descent optimization.
 * @param layer A pointer to the Layer structure containing the weights, biases, and activation function.
 * @param loss_function The loss function to be used for calculating the error during optimization.
 */

void gradient_descent(double *input_vector, double *expected_values, double learning_rate, uint32_t iterations, Layer *layer, LossFunction loss_function) {
    double *output_vector = (double *)malloc(layer->output_size * sizeof(double));
    double *deltas = (double *)malloc(layer->output_size * sizeof(double));
    double *errors = (double *)malloc(layer->output_size * sizeof(double));

    if (!output_vector || !deltas || !errors) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    double error;

    for (uint32_t iter = 0; iter < iterations; iter++) {
        // i) Forward pass
        matrix_vector_multiplication(input_vector, layer->input_size, output_vector, layer->output_size, layer->weights);

        // ii) Adding bias
        for (int i = 0; i < layer->output_size; i++) {
            output_vector[i] += layer->biases[i];
        }

        // iii) Apply the activation function
        apply_activation(output_vector, layer->output_size, layer->activation);

        // iv) Compute loss derivative with respect to output
        compute_loss_derivative(loss_function, output_vector, expected_values, errors, layer->output_size);

        // v) Calculate deltas (error * derivative of activation)
        apply_derivative(output_vector, layer->output_size, layer->activation); // Get derivatives
        for (int i = 0; i < layer->output_size; i++) {
            deltas[i] = errors[i] * output_vector[i]; // Multiply error by derivative
        }

        // vi) Update weights and biases using the deltas
        for (int i = 0; i < layer->output_size; i++) {
            for (int j = 0; j < layer->input_size; j++) {
                layer->weights[i][j] -= learning_rate * deltas[i] * input_vector[j];
            }
            layer->biases[i] -= learning_rate * deltas[i];
        }

        if (iter % 100 == 0) {
            error = compute_loss(loss_function, output_vector, expected_values, layer->output_size);
            printf("Iteration %u: Loss = %f\n", iter, error);
        }
    }

    free(output_vector);
    free(deltas);
    free(errors);
}

/**
 * Performs backpropagation through a deep neural network.
 * This function computes the gradients for weights and biases based on the 
 * loss derivative and updates the parameters using the specified optimizer and regularizer.
 * 
 * @param nn A pointer to the NeuralNetwork structure containing the layers, weights, 
 *           biases, and output vector of the network.
 * @param input_vector A pointer to the input data vector for the current training instance.
 * @param expected_values A pointer to the expected output values (ground truth) for the input vector.
 * @param learning_rate The rate at which the weights are updated during training.
 * @param optimizer A pointer to the Optimizer structure specifying the optimization strategy 
 *                  to be used for updating weights.
 * @param regularizer A pointer to the Regularizer structure to apply regularization 
 *                    techniques, if any, during the weight update.
 */

void backpropagation(NeuralNetwork *nn, double *input_vector, double *expected_values, double learning_rate, Optimizer *optimizer, Regularizer *regularizer) {
    // Perform forward pass through the network
    forward_pass(nn, input_vector);

    // Matrix for term errors
    double **deltas = (double **)malloc(nn->num_layers * sizeof(double *));
    if (!deltas) {
        fprintf(stderr, "Memory allocation failed for deltas\n");
        return;
    }

    for (int i = 0; i < nn->num_layers; i++) {
        deltas[i] = (double *)malloc(nn->layers[i].output_size * sizeof(double));
        if (!deltas[i]) {
            fprintf(stderr, "Memory allocation failed for deltas[%d]\n", i);
            for (int j = 0; j < i; j++) {
                free(deltas[j]);
            }
            free(deltas);
            return;
        }
    }

    // Backpropagation starts with calculating the error at the output layer
    Layer *output_layer = &nn->layers[nn->num_layers - 1];
    
    // Loss derivative calculation (output layer)
    compute_loss_derivative(output_layer->loss_func, nn->output_vector, expected_values, deltas[nn->num_layers - 1], output_layer->output_size);

    // Backpropagate errors for hidden layers
    for (int i = nn->num_layers - 2; i >= 0; i--) {
        Layer *current_layer = &nn->layers[i];
        Layer *next_layer = &nn->layers[i + 1];

        // Error term for the current layer
        for (int j = 0; j < current_layer->output_size; j++) {
            double weighted_error_sum = 0.0;
            for (int k = 0; k < next_layer->output_size; k++) {
                weighted_error_sum += deltas[i + 1][k] * next_layer->weights[k][j];
            }
            // Populating the matrix of deltas
            deltas[i][j] = weighted_error_sum;
        }

        // Apply the derivative of the activation function
        apply_derivative(deltas[i], current_layer->output_size, current_layer->derivative);
    }

    // Update weights and biases for each layer using the optimizer and regularizer
    for (int i = 0; i < nn->num_layers; i++) {
        Layer *current_layer = &nn->layers[i];
         //  Input from previous layer or network input
        double *input_to_layer = (i == 0) ? input_vector : nn->layers[i - 1].output_vector;

        // Update weights for each neuron
        for (int j = 0; j < current_layer->output_size; j++) {
            for (int k = 0; k < current_layer->input_size; k++) {
                // Gradient: delta * input
                double gradient = deltas[i][j] * input_to_layer[k];

                // Apply regularization if needed
                // Might explain regularization later
                if (regularizer->reg_type == L2) {
                    gradient += regularizer->lambda * current_layer->weights[j][k];
                } else if (regularizer->reg_type == L1) {
                    gradient += regularizer->lambda * ((current_layer->weights[j][k] > 0) ? 1 : -1);
                }

                // Weights are updated using the current optimizer
                update_weights(optimizer, &current_layer->weights[j][k], &gradient, 1, regularizer);
            }

            // Update bias
            double bias_gradient = deltas[i][j];
            update_weights(optimizer, &current_layer->biases[j], &bias_gradient, 1, NULL);
        }
    }

    for (int i = 0; i < nn->num_layers; i++) {
        free(deltas[i]);
    }
    free(deltas);
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

    // Parameter validation
    if (num_layers <= 0 || input_size <= 0 || output_size <= 0) {
        fprintf(stderr, "Invalid parameters: num_layers=%d, input_size=%d, output_size=%d\n",
                num_layers, input_size, output_size);
        return;
    }

    double *output_of_current_layer = (double *)malloc(layers[num_layers - 1].output_size * sizeof(double));
    if (!output_of_current_layer) {
        fprintf(stderr, "Memory allocation failed for output_of_current_layer\n");
        return;
    }

    double *input_to_next_layer = input_vector;

    // Forward pass through the layers of the neural network
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        Layer *current_layer = &layers[layer_idx];

        // Matrix-vector multiplication to compute linear transformation
        matrix_vector_multiplication(input_to_next_layer, current_layer->input_size,
                                     output_of_current_layer, current_layer->output_size,
                                     current_layer->weights);

        // Apply biases and activation function in order to introduce non-linearity
        for (int i = 0; i < current_layer->output_size; i++) {
            double z = output_of_current_layer[i] + current_layer->biases[i];
            apply_activation(&output_of_current_layer[i], 1, current_layer->activation);
        }

        // The final layer is copied to the output vector
        if (layer_idx == num_layers - 1) {
            memcpy(output_vector, output_of_current_layer, output_size * sizeof(double));
        } else {
            // Prepare for next layer
            input_to_next_layer = output_of_current_layer; 
        }
    }

    // Apply softmax for multi-class classification (PENDING)
    if (layers[num_layers - 1].activation == SOFTMAX) {
        softmax(output_vector, output_vector, output_size);
    }

    free(output_of_current_layer); 
}