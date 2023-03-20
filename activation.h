#ifndef ACTIVATION_H
#define ACTIVATION_H

#include"matrix.h"

double sigmoid(double);
double relu(double);

double sigmoid_gradient(double);
double tanh_gradient(double);
double relu_gradient(double);

#endif