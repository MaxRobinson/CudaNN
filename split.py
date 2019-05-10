import math
import os
from os import path
import csv
from random import shuffle

data = []

with open(path.join("", "", "celebrityFacesReps.txt")) as f: 
    with open("celebrityFacesGT.txt") as f2: 
        reader = csv.reader(f)
        for row in reader: 
            # rep   
            row_d = []
            for item in row: 
                row_d .append(float(item))
            #gt 
            gt = []
            line = f2.readline()
            values = line.split(",")
            for value in values:
                gt.append(float(value))

            row_d.append(gt)
            data.append(row_d)
print(data[0])

shuffle(data)

# split data
train = data[:math.floor(2*len(data)/3)]
test = data[math.floor(2*len(data)/3):]

x_train = [x[:128] for x in train]
y_train = [y[128] for y in train]

x_test = [x[:128] for x in test]
y_test = [y[128] for y in test]


with open("celebrityFacesRepsTrainShuffled.txt", 'w') as f: 
    writer = csv.writer(f)
    for rep in x_train: 
        writer.writerow(rep)

with open("celebrityFacesGTTrainShuffled.txt", 'w') as f: 
    writer = csv.writer(f)
    for rep in y_train: 
        writer.writerow(rep)

with open("celebrityFacesRepTestShuffled.txt", 'w') as f: 
    writer = csv.writer(f)
    for rep in x_test: 
        writer.writerow(rep)

with open("celebrityFacesGTTestShuffled.txt", 'w') as f: 
    writer = csv.writer(f)
    for rep in y_test: 
        writer.writerow(rep)