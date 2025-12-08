#include "MLP.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

// sigmoid
static inline double act(double x) { return 1.0 / (1.0 + exp(-x)); }
static inline double act_deriv(double y) { return y * (1.0 - y); }

// Allocate layer
static void layer_init(Layer *L, int size, int prev) {
    L->size = size;
    L->output = calloc(size, sizeof(double));
    L->z = calloc(size, sizeof(double));
    L->bias = calloc(size, sizeof(double));
    L->d_bias = calloc(size, sizeof(double));

    L->weights = calloc(prev * size, sizeof(double));
    L->d_weights = calloc(prev * size, sizeof(double));

    for (int i = 0; i < size; i++)
        L->bias[i] = (double)rand() / RAND_MAX - 0.5;

    for (int i = 0; i < prev * size; i++)
        L->weights[i] = (double)rand() / RAND_MAX - 0.5;
}

// Create MLP
MLP *mlp_create(int *sizes, int num_layers) {
    MLP *m = calloc(1, sizeof(MLP));
    m->num_layers = num_layers;
    m->sizes = malloc(num_layers * sizeof(int));
    memcpy(m->sizes, sizes, num_layers * sizeof(int));

    m->layers = calloc(num_layers, sizeof(Layer));

    for (int l = 1; l < num_layers; l++)
        layer_init(&m->layers[l], sizes[l], sizes[l - 1]);

    return m;
}

void mlp_free(MLP *m) {
    if (!m) return;
    for (int l = 1; l < m->num_layers; l++) {
        Layer *L = &m->layers[l];
        free(L->output);
        free(L->z);
        free(L->bias);
        free(L->weights);
        free(L->d_bias);
        free(L->d_weights);
    }
    free(m->layers);
    free(m->sizes);
    free(m);
}

// Forward (inference)
void mlp_forward(MLP *m, const double *input) {
    memcpy(m->layers[0].output, input, m->sizes[0] * sizeof(double));

    for (int l = 1; l < m->num_layers; l++) {
        Layer *L = &m->layers[l];
        Layer *P = &m->layers[l - 1];
        int prev = P->size;

        for (int i = 0; i < L->size; i++) {
            double sum = L->bias[i];
            for (int j = 0; j < prev; j++)
                sum += P->output[j] * L->weights[i * prev + j];
            L->output[i] = act(sum);
        }
    }
}

// Forward (training)
void mlp_forward_train(MLP *m, const double *input) {
    memcpy(m->layers[0].output, input, m->sizes[0] * sizeof(double));

    for (int l = 1; l < m->num_layers; l++) {
        Layer *L = &m->layers[l];
        Layer *P = &m->layers[l - 1];
        int prev = P->size;

        for (int i = 0; i < L->size; i++) {
            double z = L->bias[i];
            for (int j = 0; j < prev; j++)
                z += P->output[j] * L->weights[i * prev + j];
            L->z[i] = z;
            L->output[i] = act(z);
        }
    }
}

// Backprop
void mlp_backward(MLP *m, const double *target) {
    int L_last = m->num_layers - 1;
    Layer *O = &m->layers[L_last];
    int out_size = O->size;

    // output layer gradient
    for (int i = 0; i < out_size; i++) {
        double y = O->output[i];
        double d = (y - target[i]) * act_deriv(y);
        O->d_bias[i] += d;
    }

    // hidden layers
    for (int l = L_last; l > 1; l--) {
        Layer *L = &m->layers[l];
        Layer *P = &m->layers[l - 1];
        int prev = P->size;

        for (int i = 0; i < L->size; i++) {
            double d = L->d_bias[i];

            for (int j = 0; j < prev; j++) {
                L->d_weights[i * prev + j] += d * P->output[j];
                P->d_bias[j] += d * L->weights[i * prev + j] *
                                act_deriv(P->output[j]);
            }
        }
    }
}

// SGD update
void mlp_update(MLP *m, double lr) {
    for (int l = 1; l < m->num_layers; l++) {
        Layer *L = &m->layers[l];
        int prev = m->sizes[l - 1];

        for (int i = 0; i < L->size; i++) {
            L->bias[i] -= lr * L->d_bias[i];
            L->d_bias[i] = 0.0;
        }

        for (int i = 0; i < L->size * prev; i++) {
            L->weights[i] -= lr * L->d_weights[i];
            L->d_weights[i] = 0.0;
        }
    }
}

// Predict
int mlp_predict(MLP *m, const double *input) {
    mlp_forward(m, input);
    Layer *O = &m->layers[m->num_layers - 1];

    int argmax = 0;
    for (int i = 1; i < O->size; i++)
        if (O->output[i] > O->output[argmax])
            argmax = i;

    return argmax;
}

// Save / Load
int mlp_save(MLP *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;

    fwrite(&m->num_layers, sizeof(int), 1, f);
    fwrite(m->sizes, sizeof(int), m->num_layers, f);

    for (int l = 1; l < m->num_layers; l++) {
        Layer *L = &m->layers[l];
        int prev = m->sizes[l - 1];
        fwrite(L->bias, sizeof(double), L->size, f);
        fwrite(L->weights, sizeof(double), L->size * prev, f);
    }

    fclose(f);
    return 1;
}

int mlp_load(MLP *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    int nl;
    fread(&nl, sizeof(int), 1, f);

    if (nl != m->num_layers) {
        fclose(f);
        return 0;
    }

    int tmp_sizes[64];
    fread(tmp_sizes, sizeof(int), nl, f);

    for (int l = 1; l < nl; l++) {
        if (tmp_sizes[l] != m->sizes[l]) {
            fclose(f);
            return 0;
        }
    }

    for (int l = 1; l < m->num_layers; l++) {
        Layer *L = &m->layers[l];
        int prev = m->sizes[l - 1];
        fread(L->bias, sizeof(double), L->size, f);
        fread(L->weights, sizeof(double), L->size * prev, f);
    }

    fclose(f);
    return 1;
}
