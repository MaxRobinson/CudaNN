#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <cuda.h>

#include <curand.h>
#include <curand_kernel.h>

#include <cublas.h>
#include "cublas_v2.h"

#define MAX 100
#define MULTIPLIER 1.0

#define MAX_ARRAY_SIZE 1<<20

#define index(i,j,ld) (((j)*(ld))+(i))

// Struct to hold multiple timing metrics per run for comparison
struct MultTiming {
    float initTime;
    float overallTime;
};

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
__global__ void sigmoid(float* input){
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 
    if(tid < num_elements)
	{
        input[tid] = 1 / (1-exp(-1*));
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


void initWeights(float d_array, int arraySize){
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
    CURAND_CALL(curandGenerateUniform(gen, d_array, arraySize));
}

// multiplies an input vector 1 row x yColumns 
float* layerMult(float* input_values, int input_size, float * weights, int weight_row_size, int weight_col_size){
    cublasStatus status;
    cublasHandle_t handle;
    cublasCreate(&handle);
    float* layer_outputs;
    CUDA_CALL(cudaMalloc((void **) &layer_outputs, (weight_col_size) * sizeof(float));
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                1, weight_col_size, input_size, // a_rows, b_columns, a_columns
                1, // alpha
                input_values, input_size, // a, a leading dimension (height)
                weights1, weight_row_size, // b, rows of weights (= input_size)
                0, // beta 
                layer_outputs, weight_col_size); // output,  output leading dimension (height)
    status = cublasGetError();
    
    // perform sigmoid transform to hidden layer values
    // input now has hidden layer 1 values
    sigmoid<<<1, weight_col_size>>>(layer_outputs);
    return layer_outputs;
}

// Calculates the forward pass of a neural network with 2 hidden layers
// each layer is calculation of 1/(1-exp(-1*sum(x_i*w_i))) for each node in the next layer 
// where x is an input vector and w corresponds to the weights for that input node per each input value from x
float* forwardPass(float* input_values, int input_size
                float* weights1, int hidden_layer_1_size,
                float* weights2, int hidden_layer_2_size,
                float* weights3, int output_layer_size
            ){
    // wieght matrix size = input_size X hidden_layer_size
    // Weigth matrix is stored as each array of weights is a column. 
    // So the height of W (number of rows) is = to the number of nodes in the previous layer
    // The width of W (number of columns) is = to the number of nodes in the next layer
    
    // allocate the array to store the output of the input x hidden layer weigths = hidden_layer 1 values
    float* layer1_outputs = layerMult(input_values, input_size, wieghts1, input_size, hidden_layer_1_size);
    float* layer2_outputs = layerMult(layer1_outputs, hidden_layer_1_size, wieghts2, hidden_layer_1_size, hidden_layer_2_size);
    float* output = layerMult(layer2_outputs, hidden_layer_2_size, wieghts3, hidden_layer_2_size, output_layer_size);
    
    return output;
}

/**
* Main program
*
*/
int main(int argc, char** argv) {
    
    cublasInit();
    int input_layer_size = 10; 
    int hidden_layer_1_size = 10; 
    int hidden_layer_2_size = 10;
    int output_layer_size = 3; 

    // wieght matrix size = input_size X hidden_layer_size
    // Weigth matrix is stored as each array of weights is a column. 
    // So the hieght of W (number of rows) is = to the number of inputs into the next layer's node
    // The width of W (number of columns) is = to the number of nodes in the next layer
    float * input_values;
    float* wieghts1, weights2, weights3; 
    
    // cuda malloc input value space on GPU
    CUDA_CALL(cudaMalloc((void **) &input_values, (input_layer_size) * sizeof(float));

    // cuda malloc space for weight matrices on GPU
    CUDA_CALL(cudaMalloc((void **) &weights1, (input_layer_size * hidden_layer_1_size) * sizeof(float));
    CUDA_CALL(cudaMalloc((void **) &weights2, (hidden_layer_1_size * hidden_layer_2_size) * sizeof(float));
    CUDA_CALL(cudaMalloc((void **) &weights3, (hidden_layer_2_size * output_layer_size) * sizeof(float));

    // init input as random for testing for now
    initWeights(input_values, input_layer_size);
    initWeights(weights1, input_layer_size * hidden_layer_1_size);
    initWeights(weights2, hidden_layer_1_size * hidden_layer_2_size);
    initWeights(weigths3, hidden_layer_2_size * output_layer_size);

    // Weight M
    forwardPass(input_values, weights1, weights2, weights3);

    return true;
}
