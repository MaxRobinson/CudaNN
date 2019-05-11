#! /bin/bash

./network.exe --archFile arch --training celebrityFacesRepsTrainShuffled.txt --groundTruth celebrityFacesGTTrainShuffled.txt --evaluation hold --output weightsTest.txt --epochs 100 --weights weightsTest.txt
