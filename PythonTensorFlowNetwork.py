import math
import os
from os import path
import csv
from random import shuffle
import time

import tensorflow as tf
import numpy as np


# Get Data
root_path = "./celebrityFaces"
data = []

folders = os.listdir(root_path)
for folder in folders:
    reps = []
    with open(path.join(root_path, folder, "reps.txt")) as f:
        with open(path.join(root_path, folder, "gt.txt")) as f2:
            reader = csv.reader(f)
            for row in reader:
                #rep
                row_d = []
                for item in row:
                    row_d.append(float(item))
                #gt
                gt = []
                line = f2.readline()
                values = line.split(",")
                for value in values:
                    gt.append(int(value))

                row_d.append(gt)
                reps.append(row_d)
    data.extend(reps)
print(data[0])

shuffle(data)

# split data
train = data[:math.floor(2*len(data)/3)]
test = data[math.floor(2*len(data)/3):]

x_train = [x[:128] for x in train]
y_train = [y[128] for y in train]

x_test = [x[:128] for x in test]
y_test = [y[128] for y in test]

# Code derived from
# https://medium.com/@curiousily/tensorflow-for-hackers-part-ii-building-simple-neural-network-2d6779d2f91b

random_state = 42
np.random.seed(random_state)
tf.set_random_seed(random_state)

class_names = ["AC", "SJ", "Other"]

# x_train = np.random.rand(5, 146)
# y_train = np.random.rand(5, 3)
# x_test = np.random.rand(5, 146)
# y_test = np.random.rand(5, 3)

def multilayer_perceptron(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


n_input = 128
n_hidden_1 = 513
n_hidden_2 = 513
n_classes = 3

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

keep_prob = tf.placeholder("float")

training_epochs = 5000
display_step = 1000
batch_size = 1

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])



predictions = multilayer_perceptron(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cost)

cross_val_number = 3
data_len = math.floor(len(data)/cross_val_number)
data_parts = []
for i in range(cross_val_number):
    data_parts.append(data[i*data_len:(i+1)*data_len])



accuracy_list = []

for i in range(cross_val_number):
    train = []
    test = data_parts[i]
    for j, data in enumerate(data_parts):
        if i == j:
            continue
        train.extend(data)

    x_train = [x[:128] for x in train]
    y_train = [y[128] for y in train]

    x_test = [x[:128] for x in test]
    y_test = [y[128] for y in test]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_itter_time = 0
        total_time_per_run = 0
        for epoch in range(training_epochs):
            # timing
            # average_time_per_run = 0
            epoch_start_time = time.time()
            avg_cost = 0.0
            
            
            total_batch = int(len(x_train) / batch_size)
            x_batches = np.array_split(x_train, total_batch)
            y_batches = np.array_split(y_train, total_batch)

            for i in range(total_batch):
                start = time.time() # timing
                
                batch_x, batch_y = x_batches[i], y_batches[i]
                _, c = sess.run([optimizer, cost],
                                feed_dict={
                                    x: batch_x,
                                    y: batch_y,
                                    keep_prob: 0.8
                                })
                
                stop = time.time() # timing

                avg_cost += c / total_batch
                #timing
                total_time_per_run += stop-start

            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                      "{:.9f}".format(avg_cost))
            
            # timing
            epoch_stop_time = time.time()
            total_itter_time += epoch_stop_time - epoch_start_time
            
            print("Average time per epoch: {}ms".format((total_itter_time/(epoch+1)*1000)))
            print("Average time per kernel run: {}ms".format((total_time_per_run/(total_batch*(epoch+1)))*1000))
        
        print("Optimization Finished!")
        print("==============\n")
        print("Average time per epoch: {}ms".format((total_itter_time/(epoch+1)*1000)))
        print("Average time per kernel run: {}ms".format((total_time_per_run/(total_batch*(epoch+1)))*1000))
            

        
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
        print(correct_prediction)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accur_val = accuracy.eval({x: x_test, y: y_test, keep_prob: 1.0})
        print("Accuracy:", accur_val)

        accuracy_list.append(accur_val)

standard_deviation = np.std(accuracy_list)
average_accuracy = np.average(accuracy_list)
print(accuracy_list)
print(average_accuracy)
print(standard_deviation)



# v1 = tf.Variable(0.0)
# p1 = tf.placeholder(tf.float32)
# new_val = tf.add(v1, p1)
# update = tf.assign(v1, new_val)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for _ in range(5):
#         sess.run(update, feed_dict={p1: 1.0})
#     print(sess.run(v1))