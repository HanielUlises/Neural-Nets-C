#include "lin_alg.h"

// Multiply a single input by a weight to produce a single output.
static double single_in_single_out(double input, double weight) {
    return input * weight;
}

// Weighted sum of multiple inputs against corresponding weights.
double multiple_in_single_out(double* input, double* weight, int length) {
    return weighted_sum(input, weight, length);
}

// Scalar to multiple outputs (element-wise).
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
static double weighted_sum(double* input, double* weight, int length) {
    double output = 0.0;
    for (int i = 0; i < length; i++) {
        output += input[i] * weight[i];
    }
    return output;
}

// Element-wise multiplication of a scalar with each element in a vector.
static void element_wise_multiply(double input_scalar, double* weight_vector, double* output_vector, int length) {
    for (int i = 0; i < length; i++) {
        output_vector[i] = input_scalar * weight_vector[i];
    }
}

// Matrix-matrix multiplication
void matrix_matrix_multiplication(double **A, double **B, double **result, int rows_A, int cols_A, int cols_B) {
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            result[i][j] = 0;
            for (int k = 0; k < cols_A; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Transpose of a matrix
void transpose_matrix(double **input_matrix, double **output_matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output_matrix[j][i] = input_matrix[i][j];
        }
    }
}

// Dot product of two vectors
double dot_product(double *vec1, double *vec2, int length) {
    double result = 0.0;
    for (int i = 0; i < length; i++) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

// Vector addition
void vector_addition(double *vec1, double *vec2, double *result, int length) {
    for (int i = 0; i < length; i++) {
        result[i] = vec1[i] + vec2[i];
    }
}

// Vector subtraction
void vector_subtraction(double *vec1, double *vec2, double *result, int length) {
    for (int i = 0; i < length; i++) {
        result[i] = vec1[i] - vec2[i];
    }
}

// Matrix scalar multiplication
void matrix_scalar_multiplication(double **matrix, double scalar, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] *= scalar;
        }
    }
}

// Matrix-vector addition (broadcast vector to all rows)
void matrix_vector_addition(double **matrix, double *vector, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] += vector[j];
        }
    }
}

// Printing utilities
void print_matrix(double **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void print_vector(double *vector, int length) {
    for (int i = 0; i < length; i++) {
        printf("%f ", vector[i]);
    }
    printf("\n");
}
