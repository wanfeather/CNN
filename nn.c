#include"nn.h"

#include<math.h>
#include<stdlib.h>
#include<string.h>

struct _Model
{
    Layer *layer;
    Model *forward_link, *backward_link;
}

struct _Layer
{
    Mat **input, **output, ***weight, *bias, ***w_grad, *b_grad;
    Pass fp, bp;
    Act act, act_grad;
    int input_channel, output_channel;
};

void Sigmoid(Mat *input)
{
    int index;

    for(index = 0; index < input->row * input->col; index++)
        input->element[index] = sigmoid(input->element[index]);
}

void Tanh(Mat *input)
{
    int index;

    for(index = 0; index < input->row * input->col; index++)
        input->element[index] = tanh(input->element[index]);
}

void Relu(Mat *input)
{
    int index;

    for(index = 0; index < input->row * input->col; index++)
        input->element[index] = relu(input->element[index]);
}

void Softmax(Mat *input)
{
    int index;
    double sum = 0.0;

    for(index = 0; index < input->row * input->col; index++)
    {
        input->element[index] = exp(input->element[index]);
        sum += input->element[index];
    }
    for(index = 0; index < input->row * input->col; index++)
        input->element[index] /= sum;
}

void Sigmoid_gradient(Mat *input)
{
    int index;

    for(index = 0; index < input->row * input->col; index++)
        input->element[index] = sigmoid_gradient(input->element[index]);
}

void Tanh_gradient(Mat *input)
{
    int index;

    for(index = 0; index < input->row * input->col; index++)
        input->element[index] = tanh_gradient(input->element[index]);
}

void Relu_gradient(Mat *input)
{
    int index;

    for(index = 0; index < input->row * input->col; index++)
        input->element[index] = relu_gradient(input->element[index]);
}

double Cross_Entropy(Mat *output, int class)
{
    double loss = -log(output->element[class]--);

    return loss;
}

Layer *Conv(int input_size, int input_channel, int output_size, int output_channel, int kernel_size)
{
    Layer *conv_layer = (Layer *)malloc(sizeof(Layer));
    int index_1, index_2;


    conv_layer->input = (Mat **)malloc(input_channel * sizeof(Mat *));
    for(index_1 = 0; index_1 < input_channel; index_1++)
        conv_layer->input[index_1] = new_matrix(input_size, input_size);
    
    conv_layer->output = (Mat **)malloc(output_channel * sizeof(Mat *));
    for(index_1 = 0; index_1 < output_channel; index_1++)
        conv_layer->output[index_1] = new_matrix(output_size, output_size);
    
    conv_layer->weight = (Mat ***)malloc(output_channel * sizeof(Mat **));
    conv_layer->w_grad = (Mat ***)malloc(output_channel * sizeof(Mat **));
    for(index_1 = 0; index_1 < output_channel; index_1++)
    {
        conv_layer->weight[index_1] = (Mat **)malloc(input_channel * sizeof(Mat *));
        conv_layer->w_grad[index_1] = (Mat **)malloc(input_channel * sizeof(Mat *));
        for(index_2 = 0; index_2 < input_channel; index_2++)
            conv_layer->weight[index_1][index_2] = new_matrix(kernel_size, kernel_size);
            conv_layer->w_grad[index_1][index_2] = new_matrix(kernel_size, kernel_size);
    }

    conv_layer->bias = new_matrix(1, output_channel);
    conv_layer->b_grad = new_matrix(1, output_channel);

    conv_layer->fp = conv_forward;
    conv_layer->bp = conv_backward;

    conv_layer->act = Relu;
    conv_layer->act_grad = Relu_gradient;

    conv_layer->input_channel = input_channel;
    conv_layer->output_channel = output_channel;


    return conv_layer;
}

Layer *Linear(int input_size, int output_size, const char *act_select)
{
    Layer *linear_layer = (Layer *)malloc(sizeof(Layer));
    int index;


    linear_layer->input = (Mat **)malloc(sizeof(Mat *));
    linear_layer->input[0] = new_matrix(1, input_size);
    linear_layer->output = (Mat **)malloc(sizeof(Mat *));
    linear_layer->output[0] = new_matrix(1, output_size);
    linear_layer->weight = (Mat ***)malloc(sizeof(Mat **));
    linear_layer->w_grad = (Mat ***)malloc(sizeof(Mat **));
    linear_layer->weight[0] = (Mat **)malloc(sizeof(Mat *));
    linear_layer->w_grad[0] = (Mat **)malloc(sizeof(Mat *));
    linear_layer->weight[0][0] = new_matrix(input_size, output_size);
    linear_layer->w_grad[0][0] = new_matrix(input_size, output_size);
    linear_layer->bias = new_matrix(1, output_size);
    linear_layer->b_grad = new_matrix(1, output_size);
    linear_layer->fp = linear_forward;
    linear_layer->bp = linear_backward;
    linear_layer->input_channel = 1;
    linear_layer->output_channel = 1;
    if(strcmp(act_select, "Relu") == 0)
    {
        linear_layer->act = Relu;
        linear_layer->act_grad = Relu_gradient;
    }
    else if(strcmd(act_select, "Softmax") == 0)
    {
        linear_layer->act = Softmax;
        linear_layer->act_grad = NULL;
    }


    return linear_layer;
}

Layer *Pooling(int input_size, int input_channel, int output_size, int output_channel, int kernel_size)
{
    Layer *pooling_layer = (Layer *)malloc(sizeof(Layer));
    int index;


    pooling_layer->input = (Mat **)malloc(input_channel * sizeof(Mat *));
    for(index = 0; index < input_channel; index++)
        pooling_layer->input[index] = new_matrix(input_size, input_size);
    
    pooling_layer->output = (Mat **)malloc(output_channel * sizeof(Mat *));
    for(index = 0; index < output_channel; index++)
        pooling_layer->output[index] = new_matrix(output_size, output_size);

    pooling_layer->weight = (Mat ***)malloc(sizeof(Mat **));
    pooling_layer->weight[0] = (Mat **)malloc(sizeof(Mat *));
    pooling_layer->weight[0][0] = new_matrix(kernel_size, kernel_size);
    for(index = 0; index < pooling_layer->weight[0][0]->row * pooling_layer->weight[0][0]->col; index++)
        pooling_layer->weight[0][0]->element[index] = 1.0 / kernel_size / kernel_size;

    pooling_layer->w_grad = pooling_layer->bias = pooling_layer->b_grad = NULL;

    pooling_layer->fp = avgpool_forward;
    pooling_layer->bp = avgpool_backward;

    pooling_layer->act = pooling_layer->act_grad = NULL;

    pooling_layer->input_channel = input_channel;
    pooling_layer->output_channel = output_channel;


    return pooling_layer;
}

void linear_forward(Layer *linear_layer, int dont_care)
{
    Mat *temp;

    temp = matrix_product(linear_layer->input[0], linear_layer->weight[0][0]);
    copy_matrix(linear_layer->output[0], temp);
    delete_matrix(temp);
    temp = matrix_addtion(linear_layer->output[0], linear_layer->bias);
    linear_layer->act(temp);
    copy_matrix(linear_layer->output[0], temp);
    delete_matrix(temp);
}

void linear_backward(Layer *linear_layer, int sw)
{
    Mat *temp_1, *temp_2;

    copy_matrix(linear_layer->b_grad, linear_layer->output[0]);
    temp_1 = transpose(linear_layer->input[0]);
    temp_2 = matrix_product(temp_1, linear_layer->output[0]);
    copy(linear_layer->w_grad[0][0], temp_2);
    delete_matrix(temp_1);
    delete_matrix(temp_2);

    if(sw)
    {
        temp_1 = transpose(linear_layer->weight[0][0]);
        temp_2 = matrix_product(linear_layer->output[0], linear_layer->weight[0][0]);
        delete_matrix(temp_1);
        if(linear_layer->act_grad)
            linear_layer->act_grad(linear_layer->input[0]);
        temp_1 = element_product(linear_layer->input[0], temp_2);
        copy_matrix(linear_layer->input[0], temp_1);
        delete_matrix(temp_1);
        delete_matrix(temp_2);
    }
}

void conv_forward(Layer *conv_layer, int dont_care)
{
    int index_1, index_2;
    int output_row, output_col;
    int kernel_row, kernel_col;
    double value;
    Mat **temp;

    temp = (Mat **)malloc(conv_layer->input_channel * sizeof(Mat *));
    for(index_1 = 0; index_1 < conv_layer->input_channel; index_1++)
        temp[index_1] = new_matrix(conv_layer->weight[0][0]->row, conv_layer->weight[0][0]->col);

    for(output_row = 0; output_row < conv_layer->output[0]->row; output_row++)
    {
        for(output_col = 0; output_col < conv_layer->output[0]->col; output_col++)
        {
            for(index_1 = 0; index_1 < conv_layer->input_channel; index_1++)
                for(kernel_row = 0; kernel_row < temp[index_1]->row; kernel_row++)
                    for(kernel_col = 0; kernel_col < temp[index_1]->col; kernel_col++)
                    {
                        value = get_element(conv_layer->input[index_1], output_row + kernel_row, output_col + kernel_col);
                        set_element(temp[index_1], kernel_row, kernel_col, value);
                    }

            for(index_1 = 0; index_1 < conv_layer->output_channel; index_1++)
            {
                value = 0.0;
                for(index_2 = 0; index_2 < conv_layer->input_channel; index_2++)
                    value += elementwise(temp[index_2], conv_layer->weight[index_1][index_2]);
                value += get_element(conv_layer->bias, 0, index_1);
                set_element(conv_layer->output[index_1], output_row, output_col, value);
            }
        }
    }
    for(index_1 = 0; index_1 < conv_layer->output_channel; index_1++)
        conv_layer->act(conv_layer->output[index_1]);

    for(index_1 = 0; index_1 < conv_layer->input_channel; index_1++)
        delete_matrix(temp[index_1]);
    free(temp);
}

void conv_backward(Layer *conv_layer, int sw)
{
    int index_1, index_2;
    int kernel_channel, kernel_row, kernel_col;
    int input_row, input_col;
    double value;
    Mat **temp;

    temp = (Mat **)malloc(conv_layer->input_channel * sizeof(Mat *));
    for(index_1 = 0; index_1 < conv_layer->input_channel; index_1++)
        temp[index_1] = new_matrix(conv_layer->output[0]->row, conv_layer->output[0]->col);

    for(index_1 = 0; index_1 < conv_layer->output_channel; index_1++)
        for(index_2 = 0; index_2 < conv_layer->input_channel; index_2++)
            init_matrix(conv_layer->w_grad[index_1][index_2]);

    for(kernel_row = 0; kernel_row < conv_layer->w_grad[0][0]->row; kernel_row++)
        for(kernel_col = 0; kernel_col < conv_layer->w_grad[0][0]->col; kernel_col++)
        {
            for(index_1 = 0; index_1 < conv_layer->input_channel; index_1++)
                for(input_row = 0; input_row < temp[index_1]->row; input_row++)
                    for(input_col = 0; input_col < temp[index_1]->col; input_col++)
                    {
                        value = get_element(conv_layer->input[index_1], kernel_row + input_row, kernel_col + input_col);
                        set_element(temp[index_1], input_row, input_col, value);
                    }

            for(index_1 = 0; index_1 < conv_layer->output_channel; index_1++)
                for(index_2 = 0; index_2 < conv_layer->input_channel; index_2++)
                {
                    value = elementwise(temp[index_2], conv_layer->output[index_1]);
                    set_element(conv_layer->w_grad[index_1][index_2], kernel_row, kernel_col, value);
                }
        }
    for(index_1 = 0; index_1 < conv_layer->output_channel; index_1++)
    {
        value = 0.0;
        for(index_2 = 0; index_2 < conv_layer->output[index_1]->row * conv_layer->output[index_1]->col; index_2++)
            value += conv_layer->output[index_1]->element[index_2];
        set_element(conv_layer->b_grad, 0, index_1, value);
    }

    for(index_1 = 0; index_1 < conv_layer->input_channel; index_1++)
        delete_matrix(temp[index_1]);
    free(temp);


    if(sw)
    {
        temp = (Mat **)malloc(conv_layer->input_channel * sizeof(Mat *));
        for(index_1 = 0; index_1 < conv_layer->input_channel; index_1++)
        {
            Relu_gradient(conv_layer->input[index_1]);
            new_matrix(conv_layer->input[0]->row, conv_layer->input[0]->col);
            copy_matrix(temp[index_1], conv_layer->input[index_1]);
            init_matrix(conv_layer->input[index_1]);
        }

        for(index_1 = 0; index_1 < conv_layer->output_channel; index_1++)
            for(index_2 = 0; index_2 < conv_layer->input_channel; index_2++)
                for(kernel_row = 0; kernel_row < conv_layer->weight[0][0]->row; kernel_row++)
                    for(kernel_col = 0; kernel_col < conv_layer->weight[0][0]->col; kernel_col++)
                        for(input_row = 0; input_row < conv_layer->output[0]->row; input_row++)
                            for(input_col = 0; input_col < conv_layer->output[0]->col; input_col++)
                                conv_layer->input[index_2]->element[(kernel_row + input_row) * conv_layer->input[0]->col + (kernel_col + input_col)] += conv_layer->weight[index_1][index_2]->element[kernel_row * conv_layer->weight[0][0]->col + kernel_col] * conv_layer->output[index_1]->element[input_row * conv_layer->output[0]->col + input_col];
    }
}

void avgpool_forward(Layer *pooling_layer, int dc)
{
    int index;
    int input_row, input_col;
    int output_row, output_col;
    double value;
    Mat **temp;

    temp = (Mat **)malloc(pooling_layer->input_channel * sizeof(Mat *));
    for(index = 0; index < pooling_layer->input_channel; index++)
        temp[index] = new_matrix(pooling_layer->weight[0][0]->row, pooling_layer->weight[0][0]->col);

    for(output_row = 0; output_row < pooling_layer->output[0]->row; output_row++)
        for(output_col = 0; output_col < pooling_layer->output[0]->col; output_col++)
            for(index = 0; index < pooling_layer->input_channel; index++)
            {
                for(input_row = 0; input_row < temp[index]->row; input_row++)
                    for(input_col = 0; input_col < temp[index]->col; input_col++)
                    {
                        value = get_element(pooling_layer->input[index], output_row * 2 + input_row, output_col * 2 + input_col);
                        set_element(temp[index], input_row, input_col, value);
                    }
                
                value = elementwise(temp[index], pooling_layer->weight[0][0]);
                set_element(pooling_layer->output[index], output_row, output_col, value);
            }
    
    for(index = 0; index < pooling_layer->input_channel; index)
        delete_matrix(temp[index]);
    free(temp);
}

void avgpool_backward(Layer *pooling_layer, int dc)
{
    int index;
    int output_row, output_col;
    int kernel_row, kernel_col;
    double value;
    Mat *temp;

    for(index = 0; index < pooling_layer->input_channel; index++)
        init_matrix(pooling_layer->input[index]);

    for(index = 0; index < pooling_layer->input_channel; index++)
        for(output_row = 0; output_row < pooling_layer->output[0]->row; output_row++)
            for(output_col = 0; output_col < pooling_layer->output[0]->col; output_col++)
            {
                value = get_element(pooling_layer->output[index], output_row, output_col);
                temp = scale_product(pooling_layer->weight[0][0], value);
                for(kernel_row = 0; kernel_row < temp->row; kernel_row++)
                    for(kernel_col = 0; kernel_col < temp->col;kernel_col++)
                    {
                        value = get_element(temp, kernel_row, kernel_col);
                        set_element(pooling_layer->input[index], output_row * 2 + kernel_row, output_col * 2 + kernel_col);
                    }
                delete_matrix(temp);
            }
}

void forward(Model *model, Mat *input)
{
    Model *ptr;
    int index;

    copy_matrix(model->layer->input[0], input);

    for(ptr = model; ptr; ptr = ptr->forward_link)
    {
        ptr->layer->fp(ptr->layer, 0);
        for(index = 0; index < ptr->layer->output_channel; index++)
            copy_matrix(ptr->forward_link->layer->input[index], ptr->layer->output[index]);
    }
}

void backward(Model *model, )