#! /bin/bash

./network.exe --archFile arch --training celebrityFacesRepsTrainShuffled.txt --groundTruth celebrityFacesGTTrainShuffled.txt --evaluation hold --outputFile weightsTest.txt
