#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#ifndef mrobi100_network
#define mrobi100_network

#define MAX_LAYERS 4

struct NetworkArch{
    int inputLayer; 
    int layer1; 
    int layer2;
    int outputLayer; 
} typedef NetworkArch;

struct Network{
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
        bool training = false;
        bool gt = false;
        bool usePredefWeights = true;
        bool performEvalutation = false;
        std::string archFile;
        std::string weightsFile; 
        std::string trainingFile; 
        std::string gtFile;
        std::string validationFile;
        std::string evaluationFile; 
        std::string outputFile; 
        
    
        void readInputValues(int argc, char** argv){
            //check for no inputs, and output help message if so.
            checkHelp(argc);
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
                else if (!input.compare("--groundTruth"))
                {
                    this->gtFile = std::string(argv[++i]);
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
            usePredefWeights = !weightsFile.empty();
            training = !trainingFile.empty();
            gt = !gtFile.empty();
            performEvalutation = !evaluationFile.empty();
            useValidationSet = !validationFile.empty();

            if(training && !gt || gt && !training){
                cout << "Must provide both training data and associated ground truth" << endl;
                exit(EXIT_FAILURE);
            }

            if(!training && !performEvalutation){
                cout << "User must specify if training or performing evaluation or both" << endl;
                exit(EXIT_FAILURE);
            }
        }

        void checkHelp(int argc){
            if(argc <= 1){
                cout << "Usage is: ./network.exe --archFile <> --weights <optional> --training <trainingDataFile> --groundTruth <gtFile> --validation <dataFile> --evaluation <dataFileForEval> --output <networkWeightSaveFile>" << endl;
            }
        }

};


NetworkArch* readNetworkArch(InputValues* iv){
    NetworkArch* networkArch = new NetworkArch;
    string archFile = iv->archFile;
    ifstream f (archFile);
    if (!f.good()){
        cout<< "Bad Arch File" << endl;
        exit(EXIT_FAILURE);
    }    

    string layerSize; 
    getline(f, layerSize, ',');
    networkArch->inputLayer = stoi(layerSize);
    
    getline(f, layerSize, ',');
    networkArch->layer1 = stoi(layerSize);
    
    getline(f, layerSize, ',');
    networkArch->layer2 = stoi(layerSize);
    
    getline(f, layerSize); // get last layer number
    networkArch->outputLayer = stoi(layerSize);

    f.close();

    return networkArch;
};


#endif
