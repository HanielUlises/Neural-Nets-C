#ifndef MLP_H
#define MLP_H

#include <stddef.h>

typedef struct {
    int size;            // number of neurons
    double *output;      // activations
    double *z;           // pre-activations (only for training)
    double *bias;
    double *weights;     // flattened matrix: [size_prev * size]
    double *d_bias;
    double *d_weights;
} Layer;

typedef struct {
    int num_layers;
    Layer *layers;
    int *sizes;
} MLP;

// Creation / Destruction
MLP *mlp_create(int *sizes, int num_layers);
void mlp_free(MLP *m);

// Forward passes
void mlp_forward(MLP *m, const double *input);          // inference
void mlp_forward_train(MLP *m, const double *input);    // training (stores z)

// Training
void mlp_backward(MLP *m, const double *target);
void mlp_update(MLP *m, double lr);

// Prediction
int mlp_predict(MLP *m, const double *input);

// Save / Load
int mlp_save(MLP *m, const char *path);
int mlp_load(MLP *m, const char *path);

#endif
