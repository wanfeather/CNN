#include"matrix.h"

#include<stdio.h>
#include<stdlib.h>

Mat *new_matrix(int r, int c)
{
    Mat *matrix = (Mat *)malloc(sizeof(Mat));

    matrix->row = r;
    matrix->col = c;
    matrix->element = (double *)malloc(r * c * sizeof(double));
    init_matrix(matrix);

    return matrix;
}

void init_matrix(Mat *matrix)
{
    int index;

    for(index = 0; index < matrix->row * matrix->col; index++)
        matrix->element[index] = 0.0;
}

void print_matrix(Mat *matrix)
{
    int r, c, index;

    index = 0;
    for(r = 0; r < matrix->row; r++)
    {
        for(c = 0; c < matrix->col; c++)
            printf("%lf\t", matrix->element[index++]);
        printf("\n");
    }
}

Mat *matrix_product(Mat *matrix_1, Mat *matrix_2)
{
    int r, c, index;
    double value;
    Mat *output = new_matrix(matrix_1->row, matrix_2->col);

    for(r = 0; r < output->row; r++)
        for(c = 0; c < output->col; c++)
        {
            value = 0.0;
            for(index = 0; index < matrix_1->col; index++)
                value += matrix_1->element[matrix_1->col * r + index] * matrix_2->element[c + matrix_2->col * index];
            set_element(output, r, c, value);
        }
    
    return output;
}

double elementwise(Mat *matrix_1, Mat *matrix_2)
{
    int index;
    double value;

    value = 0.0;
    for(index = 0; index < matrix_1->row * matrix_1->col; index++)
        value += matrix_1->element[index] * matrix_2->element[index];

    return value;
}

Mat *transpose(Mat *matrix)
{
    int r, c, index;
    Mat *matrix_t = new_matrix(matrix->col, matrix->row);

    index = 0;
    for(c = 0; c < matrix->col; c++)
        for(r = 0; r < matrix->row; r++)
            matrix_t->element[index++] = matrix->element[r * matrix->col + c];

    return matrix_t;
}

void delete_matrix(Mat *matrix)
{
    free(matrix->element);
    free(matrix);
}

//copy_matrix(output, input)
void copy_matrix(Mat *O, Mat *I)
{
    int index;

    for(index = 0; index < I->row * I->col; index++)
        O->element[index] = I->element[index];
}

Mat *matrix_addtion(Mat *A, Mat *B)
{
    Mat *temp = new_matrix(A->row, A->col);
    int index;

    for(index = 0; index < temp->row * temp->col; index++)
        temp->element[index] = A->element[index] + B->element[index];

    return temp;
}

Mat *element_product(Mat *A, Mat *B)
{
    Mat *temp = new_matrix(A->row, A->col);
    int index;

    for(index = 0; index < temp->row * temp->col; index++)
        temp->element[index] = A->element[index] * B->element[index];

    return temp;
}

void set_element(Mat *mat, int r, int c, double value)
{
    mat->element[r * mat->col + c] = value;
}

double get_element(Mat *mat, int r, int c)
{
    return mat->element[r * mat->col + c];
}

Mat *scale_product(Mat *mat, double scale)
{
    Mat *temp = new_matrix(mat->row,  mat->col);
    int index;

    for(index = 0; index < temp->row * temp->col; index++)
        temp->element[index] = scale * mat->element[index];

    return temp;
}