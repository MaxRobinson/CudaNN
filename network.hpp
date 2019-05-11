#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector> 
#ifndef mrobi100_network
#define mrobi100_network

#define MAX_LAYERS 4
#define DEFAULT_ALPHA .1
#define DEFAULT_EPOCHS 200

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
        bool usePredefWeights = false;
        bool performEvalutation = false;
        std::string archFile;
        std::string weightsFile; 
        std::string trainingFile; 
        std::string gtFile;
        std::string validationFile;
        std::string evaluationFile; 
        std::string outputFile; 
        int epochs = -1;
        float alpha = -1;
        
    
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
                else if (!input.compare("--epochs"))
                {
                    this->epochs = stoi(std::string(argv[++i]));
                }
                else if (!input.compare("--alpha"))
                {
                    this->alpha = stof(std::string(argv[++i]));
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

            if(alpha == -1){
                alpha = DEFAULT_ALPHA;
            }

            if(epochs <= 0){
                epochs = DEFAULT_EPOCHS;
            }
        }

        void checkHelp(int argc){
            if(argc <= 1){
                cout << "Usage is: ./network.exe --archFile <> --weights <optional> --training <trainingDataFile> --groundTruth <gtFile> --validation <dataFile> --evaluation <dataFileForEval> --output <networkWeightSaveFile>" << endl;
                exit(EXIT_SUCCESS);
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

Network* readWeightsFile(string weightsFile){
    Network* network_h = new Network; 
    ifstream f (weightsFile);
    if (!f.good()){
        cout<< "Bad Weights File" << endl;
        exit(EXIT_FAILURE);
    }  

    // read the three layers of weights
    for(int i = 0; i < 3; i++){
        vector<float> values;
        string line;
        getline(f, line);
        stringstream ss(line);
        while (ss.good()){
            string floatValue;
            getline(ss, floatValue, ',');
            values.push_back(std::stof(floatValue));
        }
        switch (i)
        {
            case 0:
                network_h->w1 = values.data();
                break;
            case 1:
                network_h->w2 = values.data();
                break;
            case 2:
                network_h->w3 = values.data();
                break;
            default:
                break;
        }
    }
    f.close();
    return network_h;
}


void readData(string filePath, vector<float*>* data, int elements_per_line){
    ifstream f (filePath);
    if (!f.good()){
        cout<< "Bad File" << endl;
        exit(EXIT_FAILURE);
    }  

    // read data in
    while(f.good()){

        float* values = (float*) malloc(elements_per_line*sizeof(float));
        
        string line;
        getline(f, line);
        
        stringstream ss(line);
        for(int i = 0; i < elements_per_line; i++){
            string floatValue;
            getline(ss, floatValue, ',');
            // cout << floatValue << endl;
            if(!floatValue.compare("") || !floatValue.compare("\n")){
                free(values);
                break;
            }
            values[i] = std::stof(floatValue);
        }
        (*data).push_back(values);
    }
    f.close();
}

void writeWeights(string filePath, NetworkArch* networkArch, Network* network){
    ofstream f (filePath);
    if (!f.good()){
        cout<< "Bad file to write weights too." << endl;
        exit(EXIT_FAILURE);
    }  

    // write csv    
    int numWeights = (networkArch->inputLayer) * (networkArch->layer1);
    float* w = network->w1;
    for(int i = 0; i < numWeights; i++){
        if(i != numWeights-1){
            f << w[i] << ",";
        }else {
            f << w[i];
        }
    }
    f<<endl;

    numWeights = (networkArch->layer1) * (networkArch->layer2);
    w = network->w2;
    for(int i = 0; i < numWeights; i++){
        if(i != numWeights-1){
            f << w[i] << ",";
        }else {
            f << w[i];
        }
    }
    f<<endl;

    numWeights = (networkArch->layer2) * (networkArch->outputLayer);
    w = network->w3;
    for(int i = 0; i < numWeights; i++){
        if(i != numWeights-1){
            f << w[i] << ",";
        }else {
            f << w[i];
        }
    }
    // do not put a newline at the end of the file
    f.close();
}

#endif
