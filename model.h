#include<nn.h>

void Lenet_5(void)
{
    Conv(28, 1, 26, 8, 3);
    Conv(26, 8, 24, 8, 3);
    Pooling(24, 8, 12, 8, 2);
    Conv(12, 8, 10, 16, 3);
    Pooling(10, 16, 5, 16, 2);
    Flatten();
    Linear(400, 120, "Relu");
    Linear(120, 84, "Relu");
    Linear(84, 10, "Softmax");
}