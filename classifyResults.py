import math
import os
from os import path
import csv
from random import shuffle

import numpy as np

# Get Data
resultsFile = "ResultsFile.txt"
evalGt = "celebrityFacesGTTestShuffled.txt"

predictedResults = []
gtData = []

with open("ResultsFile.txt") as f:
    reader = csv.reader(f)
    for row in reader:
        #rep
        row_d = []
        for item in row:
            row_d.append(float(item))
        predictedResults.append(row_d)


with open(evalGt) as f:
    reader = csv.reader(f)
    for row in reader:
        #rep
        row_d = []
        for item in row:
            row_d.append(float(item))
        gtData.append(row_d)


# classify all predictedResults from Raw 
predictions=[]
for estimate in predictedResults:
    max_index = np.array(estimate).argmax()
    result = np.zeros(len(estimate))
    result[max_index] = 1
    predictions.append(result.tolist())

# calculate the error for the predictions
actuals = []
num_errors = 0
for prediction, test_item in zip(predictions, gtData):
    # if type(prediction) is tuple:
    #     actual_prediction = prediction[0]
    # else:
    #     actual_prediction = prediction[0][0]

    # actuals.append(test_item[-1])
    if prediction != test_item:
        num_errors += 1

error_rate = num_errors / len(predictions)
#error_rate, predictions, actuals
print("Error Rate is: {}".format(error_rate))
