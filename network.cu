#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
// #include <cuda.h>
#include "cublas.h"

#include <curand.h>
#include <curand_kernel.h>

#include <cublas.h>

#define MAX 100
#define MULTIPLIER 1.0

#define MAX_ARRAY_SIZE 1<<20

#define index(i,j,ld) (((j)*(ld))+(i))
// #define index(h,w,height) (((h)*(height))+(w))
// #define index(h,w,height) (((w)*(height))+(h))

// Struct to hold multiple timing metrics per run for comparison
struct MultTiming {
    float initTime;
    float overallTime;
};

//
void printMat(float*P,int uWP,int uHP){
    //printf("\n %f",P[1]);
    int i,j;
    for(i=0;i<uHP;i++){
        printf("\n");
        for(j=0;j<uWP;j++)
            printf("%f ",P[index(i,j,uHP)]);
    }
    printf("\n");
}

void printNetwork(float* dev_input, float* dev_w1, float* dev_w2, float* dev_w3,
    int input_layer_size, int hidden_layer_1_size, int hidden_layer_2_size, int output_layer_size){
    
    float *h_input = (float*)malloc(1*input_layer_size*sizeof(float));
    cublasGetMatrix(1, input_layer_size, sizeof(float), dev_input, 1, h_input, 1);
    
    float *h_w1 = (float*)malloc(input_layer_size*hidden_layer_1_size*sizeof(float));
    cublasGetMatrix(input_layer_size, hidden_layer_1_size, sizeof(float), dev_w1, input_layer_size, h_w1, input_layer_size);

    float *h_w2 = (float*)malloc(hidden_layer_1_size*hidden_layer_2_size*sizeof(float));
    cublasGetMatrix(hidden_layer_1_size, hidden_layer_2_size, sizeof(float), dev_w2, hidden_layer_1_size, h_w2, hidden_layer_1_size);
    
    float *h_w3 = (float*)malloc(hidden_layer_2_size*output_layer_size*sizeof(float));
    cublasGetMatrix(hidden_layer_2_size, output_layer_size, sizeof(float), dev_w3, hidden_layer_2_size, h_w3, hidden_layer_2_size);


    printMat(h_input, input_layer_size, 1);
    printMat(h_w1, hidden_layer_1_size, input_layer_size);
    printMat(h_w2, hidden_layer_2_size, hidden_layer_1_size);
    printMat(h_w3, output_layer_size, hidden_layer_2_size); 
}

/** 
* from: https://devtalk.nvidia.com/default/topic/524307/need-help-with-kernel-execution-parameters/
* used to wrap cuda calls with error possiblility. 
*/ 
__host__ int cudaCall(cudaError_t value, int line) {                                                                                      
    cudaError_t _m_cudaStat = value;                                                                                
    if (_m_cudaStat != cudaSuccess) {                                                                               
            printf("Error %s at line %d \n", cudaGetErrorString(_m_cudaStat), line);           
            exit(1);                                                                                                                        
    } 
    return 0;
}
#define CUDA_CALL(value) cudaCall( value, __LINE__)

/**
* modified from https://docs.nvidia.com/cuda/curand/host-api-overview.html#host-api-overview 
* used to wrap cuRAND calls with error possilbility
*/
__host__ int cuRandCall(curandStatus_t value, const char* file, int line){ 
    if( value != CURAND_STATUS_SUCCESS) {
        printf("Error at %s:%d\n",__FILE__,__LINE__);
        return EXIT_FAILURE;
    }
    return 0;
}
#define CURAND_CALL(value) cuRandCall(value, __FILE__, __LINE__)


// apply sigmoid function to a value of arrays
__global__ void sigmoid(float* input, int num_elements){
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 
    if(tid < num_elements)
	{
        input[tid] = 1 / (1-exp(-1*input[tid]));
    }
}

/**
* Main kernel to add arrays together in parallel.
*/
__global__ void add_arrays_kernel(int num_elements,
    float alpha, 
    float * array_a, 
    float * array_b)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 
    if(tid < num_elements)
    {
        array_b[tid] = alpha * array_a[tid] + array_b[tid];
    }
}


/**
* Creates a CUDA event at the current time
* Provided by grader
*
* @param None
*
* @return time The cuda event for the current time
*/
__host__
cudaEvent_t getTime(cudaStream_t stream)
{
    cudaEvent_t time;

    cudaEventCreate(&time);
    cudaEventRecord(time, stream);

    return time;
}

/**
* Helper function to init arrays on host
*
*/ 
__host__ void init_array(int ** array, int arraySize, int offset){
    int* array_actual = *array; 
    for(int i = 0; i < arraySize; i++){
        array_actual[i] = i + offset; 
    }
}


void initWeights(float ** d_array, int arraySize){
    printf("init Weights\n");
    // init arrays on the device using cuRAND
    // code adapted from https://docs.nvidia.com/cuda/curand/host-api-overview.html#host-api-overview 
    curandGenerator_t gen;
    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 0));
    /* Generate n floats on device */
    /* This code generates floats on the device thus is calling a kernel
        to perform this operation */
    CURAND_CALL(curandGenerateUniform(gen, *d_array, arraySize));
}

// multiplies an input vector 1 row x yColumns 
float* layerMult(float* input_values, int input_size, float * weights, int weight_col_size){
    cublasStatus status;
    // const float alpha = 1.0;

    float* layer_outputs;
    // CUDA_CALL(cudaMalloc((void **) &layer_outputs, (1*weight_col_size) * sizeof(float)));

    float* dev_in;
    float* dev_w;
    float* dev_out;

    float* a = 0;
    float* b = 0;
    a = (float *)malloc (1 * 5 * sizeof (*a));
    b = (float *)malloc (5 * 1 * sizeof (*b));
    
    for(int i = 0; i < 5; i++){
        a[i] = 1;
        b[i] = 1;
    }

    

    status = cublasAlloc(5, sizeof(*a), (void **) &dev_in);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (A)\n");
        exit(1);
      }
    status = cublasSetMatrix(1, 5, sizeof(*a), a, 1, dev_in, 1);
    // if (status != CUBLAS_STATUS_SUCCESS) {
    //     fprintf (stderr, "!!!! device memory allocation error (d)\n");
    //     exit(1);
    // }

    status = cublasAlloc((5), sizeof(float), (void **) &dev_w);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (b)\n");
        exit(1);
    }
    
    status=cublasSetMatrix(5,1,sizeof(float),b, 5, dev_w, 5);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (e)\n");
      exit(1);
    }

    status = cublasAlloc(1, sizeof(float),(void **) &dev_out);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (c)\n");
        exit(1);
    }


    printf("about to run Matrix \n");
    cublasSgemm('n', 'n', 
        1, 1, 5, // a_rows, b_columns, a_columns 
        1, // alpha
        dev_in, 1, // a, a rows leading dimension (height)
        dev_w, 5, // b, rows of weights (= input_size)
        0, // beta 
        dev_out, 1); // output,  output leading dimension (height)
    
    float* c = (float *)malloc (1 * 1 * sizeof (*a));
    status = cublasGetMatrix (1, 1, sizeof(*c), dev_out, 1, c, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed\n");
        // cublasFree (devPtrA);
        cublasShutdown();
        exit(1);
    }

    cublasShutdown();
    for (int j = 0; j < 1; j++) {
        for (int i = 0; i < 1; i++) {
            printf ("%7.0f", c[index(i,j,1)]);
        }
        printf ("\n");
    }
    free(a);
    free(b);
    free(c);

    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
    //             1, weight_col_size, input_size, // a_rows, b_columns, a_columns
    //             &alpha, // alpha
    //             input_values, input_size, // a, a rows leading dimension (height)
    //             weights, input_size, // b, rows of weights (= input_size)
    //             0, // beta 
    //             layer_outputs, weight_col_size); // output,  output leading dimension (height)
    // status = cublasGetError();
    
    // perform sigmoid transform to hidden layer values
    // input now has hidden layer 1 values
    // printf("about to run sigmoid\n");
    // sigmoid<<<1, weight_col_size>>>(layer_outputs, weight_col_size);
    return layer_outputs;
}

// Calculates the forward pass of a neural network with 2 hidden layers
// each layer is calculation of 1/(1-exp(-1*sum(x_i*w_i))) for each node in the next layer 
// where x is an input vector and w corresponds to the weights for that input node per each input value from x
float* forwardPass(float* input_values, int input_size,
                float* weights1, int hidden_layer_1_size,
                float* weights2, int hidden_layer_2_size,
                float* weights3, int output_layer_size
            ){
    printf("forward Pass \n");
    // wieght matrix size = input_size X hidden_layer_size
    // Weigth matrix is stored as each array of weights is a column. 
    // So the height of W (number of rows) is = to the number of nodes in the previous layer
    // The width of W (number of columns) is = to the number of nodes in the next layer
    
    // allocate the array to store the output of the input x hidden layer weigths = hidden_layer 1 values
    float* layer1_outputs = layerMult(input_values, input_size, weights1, hidden_layer_1_size);
    float* layer2_outputs = layerMult(layer1_outputs, hidden_layer_1_size, weights2, hidden_layer_2_size);
    float* output = layerMult(layer2_outputs, hidden_layer_2_size, weights3, output_layer_size);

    return output;
}

/**
* Main program
*
*/
int main(int argc, char** argv) {
    cublasStatus status;
    cublasInit();
    int input_layer_size = 3; 
    int hidden_layer_1_size = 10; 
    int hidden_layer_2_size = 5;
    int output_layer_size = 3; 

    // wieght matrix size = input_size X hidden_layer_size
    // Weigth matrix is stored as each array of weights is a column. 
    // So the hieght of W (number of rows) is = to the number of inputs into the next layer's node
    // The width of W (number of columns) is = to the number of nodes in the next layer
    float * input_values;
    float* weights1; 
    float* weights2;
    float* weights3; 
    
    // cuda malloc input value space on GPU
    CUDA_CALL(cudaMalloc((void **) &input_values, (input_layer_size) * sizeof(float)));

    // // cuda malloc space for weight matrices on GPU
    CUDA_CALL(cudaMalloc((void **) &weights1, (input_layer_size * hidden_layer_1_size) * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &weights2, (hidden_layer_1_size * hidden_layer_2_size) * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &weights3, (hidden_layer_2_size * output_layer_size) * sizeof(float)));

    // status = cublasAlloc((input_layer_size * hidden_layer_1_size), sizeof(float), (void **) &weights1);
    // status = cublasAlloc((hidden_layer_1_size * hidden_layer_2_size), sizeof(float), (void **) &weights2);
    // status = cublasAlloc((hidden_layer_2_size * output_layer_size), sizeof(float), (void **) &weights3);

    // // init input as random for testing for now
    initWeights(&input_values, input_layer_size);
    initWeights(&weights1, input_layer_size * hidden_layer_1_size);
    initWeights(&weights2, hidden_layer_1_size * hidden_layer_2_size);
    initWeights(&weights3, hidden_layer_2_size * output_layer_size);

    // float *h_input = (float*)malloc(1*input_layer_size*sizeof(float));
    // cublasGetMatrix(1, input_layer_size, sizeof(float), input_values, 1, h_input, 1);
    

    // printMat(C, input_layer_size, 1);
    // printNetwork(input_values, weights1, weights2, weights3, 
    //             input_layer_size, hidden_layer_1_size, hidden_layer_2_size, output_layer_size);

    // // Weight M
    forwardPass(input_values, input_layer_size,
        weights1, hidden_layer_1_size,
        weights2, hidden_layer_2_size,
        weights3, output_layer_size
    );

    
    return true;
}

