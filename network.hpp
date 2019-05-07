#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#ifndef network
#define network

#define MAX_LAYERS 4

struct NetworkArch{
    int inputLayer; 
    int layer1; 
    int layer2;
    int outputLayer; 
} typedef NetworkArch;

struct Network{
    float* input_values; 
    float* w1; 
    float* w2; 
    float* w3; 
} typedef Network;

struct NetworkOuput{
    float* layer1; 
    float* layer2; 
    float* output; 
} typedef NetworkOutput;


using namespace std;
class InputValues {
    public: 
        bool useValidationSet = false;
        std::string archFile;
        std::string weightsFile; 
        std::string trainingFile; 
        std::string validationFile;
        std::string evaluationFile; 
        std::string outputFile; 
        
    
        void readInputValues(int argc, char** argv){
            for(int i = 0; i < argc; i++){

                std::string input(argv[i]);

                if (!input.compare("--archFile"))
                {
                    this->archFile = std::string(argv[++i]);
                }
                else if (!input.compare("--weights"))
                {
                    this->weightsFile = std::string(argv[++i]);
                }
                else if (!input.compare("--training"))
                {
                    this->trainingFile = std::string(argv[++i]);
                }
                else if (!input.compare("--validation"))
                {
                    this->validationFile = std::string(argv[++i]);
                }
                else if (!input.compare("--evaluation"))
                {
                    this->evaluationFile = std::string(argv[++i]);
                }
                else if (!input.compare("--output"))
                {
                    this->outputFile = std::string(argv[++i]);
                }
            }
        }

        void validateArgs(){
            if(archFile.empty()){
                cout << "No arch file specified, Exiting" << endl;
                exit(EXIT_FAILURE);
            }
            if(!validationFile.empty()){
                useValidationSet = true;
            }
        }

};



#endif
