#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>

#include "cublas.h"
#include <curand.h>
#include <curand_kernel.h>

#include "network.hpp"

#define DEBUG false
#define DEBUGNET false
#define index(i,j,ld) (((j)*(ld))+(i))
#define ALPHA .1


using namespace std;

/*
*  Print Matrix on host
*
*/
void printMat(float*P,int uWP,int uHP){
    //printf("\n %f",P[1]);
    int i,j;
    for(i=0;i<uHP;i++){
        if(i != 0){
            printf("\n");
        }
        for(j=0;j<uWP;j++)
            printf("%f ",P[index(i,j,uHP)]);
            // cout << P[index(i,j,uHP)];
    }
    printf("\n");
}

/*
*  For printing entire network from device
*  meant for debugging
*/
void printNetworkFromDev(float* dev_input, float* dev_w1, float* dev_w2, float* dev_w3,
    int input_layer_size, int hidden_layer_1_size, int hidden_layer_2_size, int output_layer_size){
    
    float *h_input = (float*)malloc(1*input_layer_size*sizeof(float));
    cublasGetMatrix(1, input_layer_size, sizeof(float), dev_input, 1, h_input, 1);
    
    float *h_w1 = (float*)malloc(input_layer_size*hidden_layer_1_size*sizeof(float));
    cublasGetMatrix(input_layer_size, hidden_layer_1_size, sizeof(float), dev_w1, input_layer_size, h_w1, input_layer_size);

    float *h_w2 = (float*)malloc(hidden_layer_1_size*hidden_layer_2_size*sizeof(float));
    cublasGetMatrix(hidden_layer_1_size, hidden_layer_2_size, sizeof(float), dev_w2, hidden_layer_1_size, h_w2, hidden_layer_1_size);
    
    float *h_w3 = (float*)malloc(hidden_layer_2_size*output_layer_size*sizeof(float));
    cublasGetMatrix(hidden_layer_2_size, output_layer_size, sizeof(float), dev_w3, hidden_layer_2_size, h_w3, hidden_layer_2_size);

    cout<<"  input" <<endl;
    printMat(h_input, input_layer_size, 1);
    cout<<"  layer1" <<endl;
    printMat(h_w1, hidden_layer_1_size, input_layer_size);
    cout<<"  layer2" <<endl;
    printMat(h_w2, hidden_layer_2_size, hidden_layer_1_size);
    cout<<"  layer3" <<endl;
    printMat(h_w3, output_layer_size, hidden_layer_2_size); 
    cout<<endl;
}

/** 
* from: https://devtalk.nvidia.com/default/topic/524307/need-help-with-kernel-execution-parameters/
* used to wrap cuda calls with error possiblility. 
*/ 
__host__ int cudaCall(cudaError_t value, int line) {                                                                                      
    cudaError_t _m_cudaStat = value;                                                                                
    if (_m_cudaStat != cudaSuccess) {                                                                               
        printf("Error %s at line %d \n", cudaGetErrorString(_m_cudaStat), line);           
            printf("Error %s at line %d \n", cudaGetErrorString(_m_cudaStat), line);           
        printf("Error %s at line %d \n", cudaGetErrorString(_m_cudaStat), line);           
        exit(1);                                                                                                                        
            exit(1);                                                                                                                        
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
        exit(1);
    }
    return 0;
}
#define CURAND_CALL(value) cuRandCall(value, __FILE__, __LINE__)


__host__ int cublasCall(cublasStatus status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS Error %d at %s:%d \n", status, file, line);
        exit(1);
    }
    return 0;
}
#define CUBLAS_CALL(value) cublasCall( value, __FILE__, __LINE__)

/*
* Kernel 
* apply sigmoid function to a value of arrays
* sigmoid = (1 / (1 + e^(-input)))
*
*/

__global__ void sigmoid(float* input, int num_elements){
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 
    if(tid < num_elements)
	{
        float value = 1.0 / (1.0 + exp(-1*input[tid]));
        input[tid] = value;
    }
}

/**
* Calculates the Squred Error Pair wise for two array of elements.
* Used to provide amount of correction needed to network to back prop through the network.
*/
__global__ void squaredError(float* predicted_values, float* actual_values, float* results, int num_elements){
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 
    if(tid < num_elements){
        float value = pow(actual_values[tid] - predicted_values[tid], 2.0);
        results[tid] = value;
    }
}

/**
* formula
* \detla k = (predicted) * (1 - predicted) * error 
* error = (actual - predicted)
*/ 
__global__ 
void outputNodeDeltaK(float* predicted_values, float* actual_values, float* res_delta_k_list, int num_elements){
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 
    if(tid < num_elements){
        float predicted_value = predicted_values[tid];
        if(predicted_value == 1){
            predicted_value -= .000001;
            // printf("%f", predicted_value);
        }
        float actual_value = actual_values[tid]; 
        float delta_k =  predicted_value * (1 - predicted_value) * (actual_value - predicted_value);
        res_delta_k_list[tid] = delta_k;
    }
}

// __global__ 
// void hiddenLayerError(float* previous_layer_weights, float* layer_outputs, float* layer_errors, float* layer_deltas, int num_elements){

// }

// delta_j output output size will be equal to size of the current layer 
__global__ 
void hiddenNodeDeltaJ(float* layer_outputs, float* contribution_factors, float* res_layer_delta_js, int num_elements){
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 
    if(tid < num_elements){
        float node_output = layer_outputs[tid];
        if(node_output == 1){
            node_output -= .000001;
            // printf("%f", node_output);
        }
        float contribution_factor = contribution_factors[tid];
        float delta_j = (1 - node_output) * node_output * contribution_factor;
        res_layer_delta_js[tid] = delta_j;
    }

}


// used for both hidden and output layer weights
// we are itterating based on wieght updates to weights all going to the same node.
__global__
void weightUpdate(float* current_weights, float* delta, float* previous_layer_input, float alpha, int offset, int num_elements){
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 
    float actual_delta = *delta;
    
    if(tid < num_elements){
        // note previous layer input array needs to be made of equal size to the weights they affect (or needs to be made a constant and this code restructured)
        // previous layer is closer to the start of the network
        current_weights[offset + tid] = (current_weights[offset+tid]) + (alpha * actual_delta * previous_layer_input[tid]);
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

/*
* initialize a GPU array using CURAND
* Currently it uses a default seed for repeatability and testing 
*
*/
void initWeights(float ** d_array, int arraySize){
    #if DEBUG
    printf("init Weights\n");
    #endif
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

/* 
* Multiplies an input vector 1 row x yColumns 
*/
float* layerMult(float* input_values, int input_size, 
                float * weights, int weight_col_size){
    float* layer_outputs;
    CUBLAS_CALL(cublasAlloc((1*weight_col_size), sizeof(float),(void **) &layer_outputs));


    cublasSgemm('n', 'n', 
        1, weight_col_size, input_size, // a_rows, b_columns, a_columns 
        1, // alpha
        input_values, 1, // a, a rows leading dimension (height)
        weights, input_size, // b, rows of weights (= input_size)
        0, // beta 
        layer_outputs, 1); // output,  output leading dimension (height)
    
    
    // perform sigmoid transform to hidden layer values
    // input now has hidden layer 1 values
    // printf("about to run sigmoid\n");
    sigmoid<<<1, weight_col_size>>>(layer_outputs, weight_col_size);
    
    #if DEBUG
    float* c = (float *)malloc (1 * weight_col_size * sizeof (float));
    int status = cublasGetMatrix (1, weight_col_size, sizeof(*c), layer_outputs, 1, c, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed\n");
        exit(1);
    }

    for (int j = 0; j < 1; j++) {
        for (int i = 0; i < weight_col_size; i++) {
            printf("%f ",c[index(i,j,1)]);
        }
        printf ("\n");
    }
    free(c);
    #endif

    return layer_outputs;
}

// Calculates the forward pass of a neural network with 2 hidden layers
// each layer is calculation of 1/(1-exp(-1*sum(x_i*w_i))) for each node in the next layer 
// where x is an input vector and w corresponds to the weights for that input node per each input value from x
NetworkOutput* forwardPass(float* input_values, int input_size,
                float* weights1, int hidden_layer_1_size,
                float* weights2, int hidden_layer_2_size,
                float* weights3, int output_layer_size
            ){
    #if DEBUG
    printf("Forward Pass \n");
    #endif
    
    // wieght matrix size = input_size X hidden_layer_size
    // Weigth matrix is stored as each array of weights is a column. 
    // So the height of W (number of rows) is = to the number of nodes in the previous layer
    // The width of W (number of columns) is = to the number of nodes in the next layer
    
    // allocate the array to store the output of the input x hidden layer weigths = hidden_layer 1 values
    float* layer1_outputs = layerMult(input_values, input_size, weights1, hidden_layer_1_size);
    float* layer2_outputs = layerMult(layer1_outputs, hidden_layer_1_size, weights2, hidden_layer_2_size);
    float* output = layerMult(layer2_outputs, hidden_layer_2_size, weights3, output_layer_size);
    
    // need to save the layer outputs
    NetworkOutput* networkOutput = new NetworkOutput; 
    
    // dont forget to cublasFree these pointers
    networkOutput -> layer1 = layer1_outputs;
    networkOutput -> layer2 = layer2_outputs;
    networkOutput -> output = output;

    return networkOutput;
}


float* calculateContributionsToError(int hidden_layer_size, int next_layer_size, float* weight_matrix_d, float* delta_ks_d){
    // get contributions to error per node for layer 2
    float* contributionsToError = (float*) malloc(hidden_layer_size*sizeof(float)); 
    for(int j = 0; j < hidden_layer_size; j++){
        // i is the index into the node the in layer 2 
        // construct the w_jk vector
        int incrx =  (hidden_layer_size);
        // this is a horizontal (row) slice out of our column based weight matrix 
        // this gives us all the weights from a single node to the other nodes it's attached to
        // i.e. from node j to node k
        // contribution error = sum(delta_k * w_jk) 
        // provides the amount this node contributed to the error in node outputs after it.
        float contributionToError = cublasSdot(next_layer_size, &weight_matrix_d[j], incrx , delta_ks_d, 1);
        #if DEBUG
        cout<<"Contribution to Error: " << contributionToError << endl;
        #endif
        contributionsToError[j] = contributionToError;
    }
    // move contributions to error to device so that we can run our delta_j kernel
    float* contribsToError_d; 
    CUBLAS_CALL(cublasAlloc(hidden_layer_size, sizeof(float), (void**) &contribsToError_d));
    CUDA_CALL(cudaMemcpy(contribsToError_d, contributionsToError, hidden_layer_size*sizeof(float),cudaMemcpyHostToDevice));
    free(contributionsToError);

    return contribsToError_d;
}

void backPropagate(NetworkArch* networkArch, Network* network, NetworkOutput* dev_network_output, float* input_values, float* actual_output_d){
    int input_layer_size = networkArch->inputLayer; 
    int hidden_layer_1_size = networkArch->layer1; 
    int hidden_layer_2_size = networkArch->layer2;
    int output_layer_size = networkArch->outputLayer; 
    
    float* weights1_d = network->w1;
    float* weights2_d = network->w2;
    float* weights3_d = network->w3;

    float* delta_ks_d;
    CUBLAS_CALL(cublasAlloc(output_layer_size, sizeof(float), (void**) &delta_ks_d));
    // start backprop
    // for fun, get the squared error for the nodes
    outputNodeDeltaK<<<1, output_layer_size>>>(dev_network_output->output, actual_output_d, delta_ks_d, output_layer_size);

    // #if DEBUG
    float* h_delta_k = (float *)malloc(output_layer_size*sizeof(float));
    CUDA_CALL(cudaMemcpy(h_delta_k, delta_ks_d, output_layer_size*sizeof(float), cudaMemcpyDeviceToHost));
    cout<<"output delta_ks"<< endl;
    printMat(h_delta_k, output_layer_size, 1);
    free(h_delta_k);
    // #endif

    // get contributions to error per node for layer 2   
    float* contribsToError_d = calculateContributionsToError(hidden_layer_2_size, output_layer_size, weights3_d, delta_ks_d);

    // run delta j Kernel
    float* delta_js_l2_d; 
    CUBLAS_CALL(cublasAlloc(hidden_layer_2_size, sizeof(float), (void**)&delta_js_l2_d));
    // Calculate the detla JS for layer2
    hiddenNodeDeltaJ<<<1, hidden_layer_2_size>>>(dev_network_output->layer2, contribsToError_d, delta_js_l2_d, hidden_layer_2_size);

    #if DEBUG
    float* h_delta_j = (float *)malloc(hidden_layer_2_size*sizeof(float));
    CUDA_CALL(cudaMemcpy(h_delta_j, delta_js_l2_d, hidden_layer_2_size*sizeof(float), cudaMemcpyDeviceToHost));
    cout<<"output delta_js"<< endl;
    printMat(h_delta_j, hidden_layer_2_size, 1);
    free(h_delta_j);
    #endif
    
    // get contributions to error per node for layer 1
    float* contribsToError_1_d = calculateContributionsToError(hidden_layer_1_size, hidden_layer_2_size, weights2_d, delta_js_l2_d);

    // run delta j Kernel
    float* delta_js_l1_d; 
    CUBLAS_CALL(cublasAlloc(hidden_layer_1_size, sizeof(float), (void**)&delta_js_l1_d));
    // Calculate the detla JS for layer1
    hiddenNodeDeltaJ<<<1, hidden_layer_1_size>>>(dev_network_output->layer1, contribsToError_1_d, delta_js_l1_d, hidden_layer_1_size);

    #if DEBUG
    float* h_delta_j_1 = (float *)malloc(hidden_layer_1_size*sizeof(float));
    CUDA_CALL(cudaMemcpy(h_delta_j_1, delta_js_l1_d, hidden_layer_1_size*sizeof(float), cudaMemcpyDeviceToHost));
    cout<<"output delta_js_1"<< endl;
    printMat(h_delta_j_1, hidden_layer_1_size, 1);
    free(h_delta_j_1);
    #endif

    

    // can now update my weight matrices. 
    // update each column in the matrix one at a time, aka. itterate based on weights all going to the same
    // node thus using 1 delta at a time for each node. 
    for(int i = 0; i < hidden_layer_1_size; i++){
        weightUpdate<<<1, input_layer_size>>>(weights1_d, &delta_js_l1_d[i], input_values, ALPHA, i * input_layer_size, input_layer_size);
    }

    for(int i = 0; i < hidden_layer_2_size; i++){
        weightUpdate<<<1, hidden_layer_1_size>>>(weights2_d, &delta_js_l2_d[i], dev_network_output->layer1, ALPHA, i * hidden_layer_1_size, hidden_layer_1_size);
    }

    for(int i = 0; i < output_layer_size; i++){
        weightUpdate<<<1, hidden_layer_2_size>>>(weights3_d, &delta_ks_d[i], dev_network_output->layer2, ALPHA, i * hidden_layer_2_size, hidden_layer_2_size);
    }

    cudaEvent_t stop = getTime(0);
    // cudaThreadSynchronize();
    cudaEventSynchronize(stop);
    cudaEventDestroy(stop);
}

// void trainNetwork(){
//     // output is still on device
//     NetworkOutput* dev_network_output = forwardPass(input_values_d, input_layer_size,
//         weights1_d, hidden_layer_1_size,
//         weights2_d, hidden_layer_2_size,
//         weights3_d, output_layer_size
//     );

//     // if training:
//     float* actual_values_d;
//     CUDA_CALL(cudaMalloc((void**)&actual_values_d, output_layer_size*sizeof(float)));
//     CUDA_CALL(cudaMemcpy(actual_values_d, <<<<>>>>>, output_layer_size*sizeof(float), cudaMemcpyHostToDevice));
    
//     // if training
//     if(iv.training){
//         backPropagate(networkArch, network, dev_network_output, input_values_d, actual_values_d);
//         #if DEBUG
//         cout<<"printing new weights"<< endl;
//         printNetworkFromDev(input_values_d, weights1_d, weights2_d, weights3_d, 
//                     input_layer_size, hidden_layer_1_size, hidden_layer_2_size, output_layer_size);
//         #endif
//     }
// }

/**
* Main program
*
*/
int main(int argc, char** argv) {

    // read CLI args
    InputValues iv = InputValues();
    iv.readInputValues(argc, argv);
    iv.validateArgs();
    int epochs = 100;
    
    NetworkArch* networkArch = readNetworkArch(&iv);
    printf("Network Arch = %d:%d:%d:%d \n", networkArch->inputLayer, networkArch->layer1, networkArch->layer2, networkArch->outputLayer);
    
    Network* network_h;
    // if weights are defined read them in
    if(iv.usePredefWeights){
        network_h = readWeightsFile(iv.weightsFile);
    }

    // if training read training data and gt
    vector<float*> trainingData_h;
    vector<float*> gtData_h;
    if(iv.training){
        // read training data
        readData(iv.trainingFile, &trainingData_h, networkArch->inputLayer);
        // read GT data
        readData(iv.gtFile, &gtData_h, networkArch->outputLayer);
    }
    #if DEBUG
    cout << "Size of training Data: " << trainingData_h.size() << endl;
    cout << "Size value of GT Data: " << trainingData_h.size() << endl;
    printf("First value of training Data: %f \n", trainingData_h[0][0]);
    printf("First value of GT Data: %f \n", trainingData_h[0][0]);
    #endif 

    // if evaluation set supplied read in

    // set outputfile


    // cublasStatus status;
    cublasInit();
    int input_layer_size = networkArch->inputLayer; 
    int hidden_layer_1_size = networkArch->layer1; 
    int hidden_layer_2_size = networkArch->layer2;
    int output_layer_size = networkArch->outputLayer; 

    // wieght matrix size = input_size X hidden_layer_size
    // Weigth matrix is stored as each array of weights is a column. 
    // So the hieght of W (number of rows) is = to the number of inputs into the next layer's node
    // The width of W (number of columns) is = to the number of nodes in the next layer
    float * input_values_d;
    float* weights1_d; 
    float* weights2_d;
    float* weights3_d; 
    
    // cuda malloc input value space on GPU
    CUDA_CALL(cudaMalloc((void **) &input_values_d, (input_layer_size) * sizeof(float)));

    // // cuda malloc space for weight matrices on GPU
    // CUDA_CALL(cudaMalloc((void **) &weights1, (input_layer_size * hidden_layer_1_size) * sizeof(float)));
    // CUDA_CALL(cudaMalloc((void **) &weights2, (hidden_layer_1_size * hidden_layer_2_size) * sizeof(float)));
    // CUDA_CALL(cudaMalloc((void **) &weights3, (hidden_layer_2_size * output_layer_size) * sizeof(float)));

    CUBLAS_CALL(cublasAlloc((input_layer_size * hidden_layer_1_size), sizeof(float), (void **) &weights1_d));
    CUBLAS_CALL(cublasAlloc((hidden_layer_1_size * hidden_layer_2_size), sizeof(float), (void **) &weights2_d));
    CUBLAS_CALL(cublasAlloc((hidden_layer_2_size * output_layer_size), sizeof(float), (void **) &weights3_d));

    Network* network = new Network;
    network->w1 = weights1_d; 
    network->w2 = weights2_d; 
    network->w3 = weights3_d; 

    // init input as random for testing for now
    if(iv.weightsFile.empty()){
        cout << "Initializing weights with Random values" << endl;

        initWeights(&input_values_d, input_layer_size);
        initWeights(&weights1_d, input_layer_size * hidden_layer_1_size);
        initWeights(&weights2_d, hidden_layer_1_size * hidden_layer_2_size);
        initWeights(&weights3_d, hidden_layer_2_size * output_layer_size);
    }

    #if DEBUGNET
    cout<< "Current Network Weights: " << endl;
    printNetworkFromDev(input_values_d, weights1_d, weights2_d, weights3_d, 
                input_layer_size, hidden_layer_1_size, hidden_layer_2_size, output_layer_size);
    #endif


    // run the network for all the data as if training
    // float* gt_d;
    // CUDA_CALL(cudaMalloc((void**)&gt_d, output_layer_size*sizeof(float)));
    // for(int i = 0; i < 1; i++){

    //     for(int i = 0; i < 1; i++){
    //         // train the network for the entire data

    //         // put input on the GPU
    //         CUDA_CALL(cudaMemcpy(input_values_d, trainingData_h[i], input_layer_size*sizeof(float), cudaMemcpyHostToDevice));
            
    //         // get network output
    //         // output is still on device
    //         NetworkOutput* dev_network_output = forwardPass(input_values_d, input_layer_size,
    //             weights1_d, hidden_layer_1_size,
    //             weights2_d, hidden_layer_2_size,
    //             weights3_d, output_layer_size
    //         );

    //         CUDA_CALL(cudaMemcpy(gt_d, gtData_h[i], output_layer_size*sizeof(float), cudaMemcpyHostToDevice));

            
            
    //         #if DEBUG
    //         // print network output
    //         float* h_output = (float *)malloc (1 * output_layer_size * sizeof (float));
    //         CUBLAS_CALL(cublasGetMatrix (1, output_layer_size, sizeof(*h_output), dev_network_output->output, 1, h_output, 1));

    //         printf("Network output: ");
    //         printMat(h_output, output_layer_size, 1);
    //         free(h_output);
    //         #endif


    //         // backProp to train the network
    //         backPropagate(networkArch, network, dev_network_output, input_values_d, gt_d);
            
    //         #if DEBUGNET
    //         cout<<"printing new weights"<< endl;
    //         printNetworkFromDev(input_values_d, weights1_d, weights2_d, weights3_d, 
    //                     input_layer_size, hidden_layer_1_size, hidden_layer_2_size, output_layer_size);
    //         #endif
    //         cudaThreadSynchronize();

    //         cublasFree(dev_network_output->layer1);
    //         cublasFree(dev_network_output->layer2);
    //         cublasFree(dev_network_output->output);
    //     }
    // }



    // output is still on device
    NetworkOutput* dev_network_output = forwardPass(input_values_d, input_layer_size,
        weights1_d, hidden_layer_1_size,
        weights2_d, hidden_layer_2_size,
        weights3_d, output_layer_size
    );

    // if training:
    float actual_value_h[3] = {1,0,0};
    float* actual_values_d;
    CUDA_CALL(cudaMalloc((void**)&actual_values_d, output_layer_size*sizeof(float)));
    CUDA_CALL(cudaMemcpy(actual_values_d, actual_value_h, output_layer_size*sizeof(float), cudaMemcpyHostToDevice));
    
    // if training
    if(iv.training){
        backPropagate(networkArch, network, dev_network_output, input_values_d, actual_values_d);
        #if DEBUGNET
        cout<<"printing new weights"<< endl;
        printNetworkFromDev(input_values_d, weights1_d, weights2_d, weights3_d, 
                    input_layer_size, hidden_layer_1_size, hidden_layer_2_size, output_layer_size);
        #endif
    }
    



    





    float* h_output = (float *)malloc (1 * output_layer_size * sizeof (float));
    CUBLAS_CALL(cublasGetMatrix (1, output_layer_size, sizeof(*h_output), dev_network_output->output, 1, h_output, 1));

    printf("Network output: ");
    printMat(h_output, output_layer_size, 1);
    free(h_output);

    cublasFree(input_values_d);
    cublasFree(weights1_d);
    cublasFree(weights2_d);
    cublasFree(weights3_d);

    cublasFree(dev_network_output->layer1);
    cublasFree(dev_network_output->layer2);
    cublasFree(dev_network_output->output);

    cublasShutdown();
    return true;
}

