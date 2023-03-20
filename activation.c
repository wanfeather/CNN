#include"activation.h"

#include<math.h>

double sigmoid(double input)
{
    return 1.0 / (1.0 + exp(-input));
}

double relu(double input)
{
    return input * (input > 0);
}

double sigmoid_gradient(double input)
{
    return input * (1.0 - input);
}

double tanh_gradient(double input)
{
    return 1.0 - input * input;
}

double relu_gradient(double input)
{
    return 1.0 * (input > 0);
}