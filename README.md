# CudaNN - Class Project for Intro to GPU Programing JHU
This project is a GPU implementation of a Neural Network with 2 hidden layers, also known as a Multi Layer Perceptron (MLP). 


The implemenation leverages CUBLAS to help with the matrix multiplication parts of the neural networks. The completed program will implement a neural network that can be trained using the [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) algorithm. The goal is to do both the forward pass and backpropogation on the GPU to enable parallelism where possible and increase the speed of training and classification. 

Code also at [https://github.com/MaxRobinson/CudaNN](https://github.com/MaxRobinson/CudaNN)

## Usage
``` bash
Usage is: ./network.exe --archFile <> --weights <optional> --training <trainingDataFile> --groundTruth <gtFile> --evaluation <dataFileForEval> --output <networkWeightSaveFile> --alpha <.1> --epochs <200>
```

To quickly see how the program works, three convenience scripts are supplied. 
* `run.sh` runs the network with the supplied `arch` file, loads in weights from `weightsTest.txt`, loads training data, trains, and then writes the weights back to `weightsTest.txt`. 
* `runWithoutWeights.sh` runs the program without any specified weights file. 
* `eval.sh` runs the program with only the evaluation set of data and loads weights from the `weightsTest.txt` file. 

## Compilation
Run `make` in the main directory. 
__NOTE:__ Ensure that the `nvcc` compiler is in your path.   
If not, run something like the following before running make `export PATH=$PATH:/usr/local/cuda-8.0/bin`  
This assumes you have CUDA installed. 
