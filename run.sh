#! /bin/bash

./network.exe --archFile arch --training celebrityFacesRepsTrainShuffled.txt --groundTruth celebrityFacesGTTrainShuffled.txt --output weightsTest.txt --epochs 100 --weights weightsTest.txt
