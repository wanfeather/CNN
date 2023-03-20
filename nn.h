#ifndef NN_H
#define NN_H

#include"matrix.h"
#include"activation.h"

typedef struct _Model Model;
typedef struct _Layer Layer;
typedef void (*Pass)(Layer *, int);
typedef void (*Act)(Mat *);


void Sigmoid(Mat *);
void Tanh(Mat *);
void Relu(Mat *);
void Softmax(Mat *);

void Sigmoid_gradient(Mat *);
void Tanh_gradient(Mat *);
void Relu_gradient(Mat *);


double Cross_Entropy(Mat *, int);


Layer *Conv(int, int, int, int, int);
Layer *Linear(int, int, const char *);
Layer *Pooling(int, int, int, int, int);
Layer *Flatten(void);


void linear_forward(Layer *, int);
void linear_backward(Layer *, int);
void conv_forward(Layer *, int);
void conv_backward(Layer *, int);
void avgpool_forward(Layer *, int);
void avgpool_backward(Layer *, int);
void flatten(Layer *, int);
void flatten_back(Layer *, int);


void forward(Model *, Mat *);
void backward(void);
void update(void);

#endif