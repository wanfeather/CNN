#include"optimizer.h"

void SGD(Layer *layer, Optim_param par)
{
    int index;

    for(index = 0; index < layer->weight->row * layer->weight->col; index++)
        layer->weight->element[index] -= par.learning_rate * layer->weight_gradient->element[index];
    if(layer->bias)
        for(index = 0; index < layer->bias->row * layer->bias->col; index++)
            layer->bias->element[index] -= par.learning_rate * layer->bias_gradient->element[index];
}