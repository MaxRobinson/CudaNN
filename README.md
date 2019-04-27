# CudaNN - Class Project for Intro to GPU Programing JHU

This project is a GPU implementation of a Neural Network with 2 hidden layers, also known as a Multi Layer Perceptron (MLP). 


The implemenation leverages CUBLAS to help with the matrix multiplication parts of the neural networks. The completed program will implement a neural network that can be trained using the [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) algorithm. The goal is to do both the forward pass and backpropogation on the GPU to enable parallelism where possible and increase the speed of training and classification. 


As of (4/26/19) the code is a work in progress. The main parts of the code are the `layerMult()`, `sigmoid()`, and `forwardPass()` functions. These functions set up and execute the forward pass of inputs through a network. The `sigmoid()` function is a custom kernel that simply applies a sigmoid activation function to each value in the array. The backpropogation portion of the algorithm has not been implemented yet. 

For right now, testing is just being done with repeatable randomly generated input as well as repeatable randomly generated values for weights. 
