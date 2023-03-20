#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include"model.h"

typedef struct _Optim_param
{
    double learning_rate, momentum, weight_decay;
}Optim_param;

void SGD(Layer *, Optim_param);
void Adagrad(Layer *, Optim_param);
void RMSprop(Layer *, Optim_param);
void Adam(Layer *, Optim_param);

#endif