#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Weighted sum of multiple inputs against corresponding weights.
double multiple_in_single_out(double* input, double* weight, int length);

// Scalar to multiple outputs (element-wise).
void single_in_multiple_out(double scalar, double* w_vect, double* out_vect, int length);

// Computes the output vector from an input vector and a matrix of weights.
void multiple_in_multiple_out(double *input_vector, int INPUT_LEN, double *output_vector, int OUTPUT_LEN, double **weight_matrix);

// Matrix-vector multiplication
void matrix_vector_multiplication(double *input_vector, int INPUT_LEN, double *output_vector, int OUTPUT_LEN, double **weight_matrix);

// Matrix-matrix multiplication
void matrix_matrix_multiplication(double **A, double **B, double **result, int rows_A, int cols_A, int cols_B);

// Transpose of a matrix
void transpose_matrix(double **input_matrix, double **output_matrix, int rows, int cols);

// Dot product of two vectors
double dot_product(double *vec1, double *vec2, int length);

// Vector addition
void vector_addition(double *vec1, double *vec2, double *result, int length);

// Vector subtraction
void vector_subtraction(double *vec1, double *vec2, double *result, int length);

// Matrix scalar multiplication
void matrix_scalar_multiplication(double **matrix, double scalar, int rows, int cols);

// Matrix-vector addition (broadcast vector to all rows)
void matrix_vector_addition(double **matrix, double *vector, int rows, int cols);

// Printing utilities (debug)
void print_matrix(double **matrix, int rows, int cols);
void print_vector(double *vector, int length);


double weighted_sum(double* input, double* weight, int length);
void element_wise_multiply(double input_scalar, double* weight_vector, double* output_vector, int length);