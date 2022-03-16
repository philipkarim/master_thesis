import torch
import torch.nn as nn
import torch.nn.functional as F

class Feedforward(nn.Module):
    """
    Neural network class to use in context of Hamiltonian parameters
    """
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

def weights_init_xavier(m):
    """
    Weight function to initialize the weights according to the xavier rule
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        #if m.bias.data is not None:
            #nn.init.xavier_uniform_(m.bias.data)



import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

class MLP2(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32 * 32 * 3, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
        )


    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)
    
""" 
    
    if __name__ == '__main__':
    
    # Set fixed random number seed
    torch.manual_seed(42)
    
    # Prepare CIFAR-10 dataset
    dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
    
    # Initialize the MLP
    mlp = MLP()
    
    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    
    # Run the training loop
    for epoch in range(0, 5): # 5 epochs at maximum
        
        # Print epoch
        print(f'Starting epoch {epoch+1}')
        
        # Set current loss value
        current_loss = 0.0
        
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
        
        # Get inputs
        inputs, targets = data
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        outputs = mlp(inputs)
        
        # Compute loss
        loss = loss_function(outputs, targets)
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        
        # Print statistics
        current_loss += loss.item()
        if i % 500 == 499:
            print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 500))
            current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')
    #Testing the neural network:

"""

import sys, math
import random
import numpy as np
import autograd.numpy as np
from autograd import grad

# Multilayer Perceptron implementation with autograd
class MLP:
    # NETWORK DIMENSIONS:
    input_layer_size = 3
    hidden_layer_size = 3
    output_layer_size = 2 # output layer / softmax layer number of neurons

    # CONSTRUCTOR:
    def __init__(self, learning_examples_array):
        self.learning_examples_array = learning_examples_array
        self.init_weights()

    def init_weights(self):
        # the list of all weights:
        self.weights = [None, None]
        # first part, between input layer and hidden layer:
        self.weights[0] = np.array([
            # Input layer to hidden layer
            # i1 connections
            [random.random(), random.random(), random.random()],
            # i2 connections
            [random.random(), random.random(), random.random()],
            # i3 connections
            [random.random(), random.random(), random.random()],
        ])

        # second part, between hidden layer and output layer:
        self.weights[1] = np.array([
            # Hidden layer to output layer
            # h1 connections
            [random.random(), random.random()],
            # h2 connections
            [random.random(), random.random()],
            # h3 connections
            [random.random(), random.random()]
        ])

        # FORWARD FEED / PASS - NODE VALUES:
        # important: use 0.0 instead of 0 (otherwise array dtype will be int)

        # weights_gradients structure follows the one used by self.weights.
        self.weights_gradients = [None, None, None]
        self.weights_gradients[0] = np.array([
            # Input layer to hidden layer
            # i1 connections
            [0.0, 0.0, 0.0],
            # i2 connections
            [0.0, 0.0, 0.0],
            # i3 connections
            [0.0, 0.0, 0.0],
        ])
        self.weights_gradients[1] = np.array([
            # Hidden layer to output layer
            # h1 connections
            [0.0, 0.0],
            # h2 connections
            [0.0, 0.0],
            # h3 connections
            [0.0, 0.0]
        ])
        
        #dL/theta, will fill this myself
        self.weights_gradients[2] = np.array([
            # Hidden layer to output layer
            # h1 connections
            [0.0],
            # h2 connections
            [0.0],
            # h3 connections
            [0.0]
        ])

    # logistic function, needed for hidden layer value calculations:
    def activation_function(self,row):
        # return 1.0 / (1.0 + math.exp(-x))
        return 1.0 / (1.0 + np.exp(-row))

    def cross_entropy(self,target_distribution, predicted_distribution):
        return -(np.dot(np.array(target_distribution), np.log(predicted_distribution)))

    # return softmax value for a specified output position:
    """
    def outputs_softmax_loss(self, outputs, correct_output_row):
        total = np.sum(np.exp(outputs))
        singles = np.exp(outputs)
        softmax_row = singles * (1.00 / total)
        # print("SOFTMAX ROW:")
        # print(softmax_row)
        self.prediction_outputs = softmax_row
        # loss_row = (correct_output_row - softmax_row) ** 2
        loss_row = self.cross_entropy(correct_output_row, softmax_row)
        return np.sum(loss_row)
    """
    # PRE: inputs row needs to be filled
    # PRE: correct_outputs row needs to be filled
    def predict(self, weights):
        matmul_hidden = np.matmul(self.inputs, weights[0])
        hidden_layer = self.activation_function(matmul_hidden)
        outputs = np.matmul(hidden_layer, weights[1])
        
        #Replace last layer with the actual gradients

        return outputs

    def make_prediction(self, image_object):
        self.inputs = np.array(image_object[1:4], dtype=float)
        self.predict(self.weights)
        return self.prediction_outputs

    def bp_update_weights(self):
        # fixed constant; speed of convergence:
        learning_rate = 0.2

        for layer in range(0, 2):
            for i in range(self.weights[layer].shape[0]):
                for j in range(self.weights[layer].shape[1]):
                    gradient_value = self.weights_gradients[layer][i][j]

                    # move up or down:
                    if (gradient_value > 0):
                        self.weights[layer][i][j] += -learning_rate * abs(gradient_value)
                        # self.weights[layer][i][j] += -learning_rate
                    elif (gradient_value < 0):
                        self.weights[layer][i][j] += learning_rate * abs(gradient_value)
                        # self.weights[layer][i][j] += learning_rate

    # CORE API:
    # take the training input data and update the weights (train the network):
    def train_network(self):
        print('Training network...')

        compute_gradients = grad(self.predict)
        total_loss = 0

        for i in range(0, self.learning_examples_array.shape[0]):
            # real probabilities (target output) for the current training example:
            target_distribution = np.array([self.learning_examples_array[i][4], self.learning_examples_array[i][5]])

            self.inputs = np.array(self.learning_examples_array[i][1:4], dtype=float)
            self.correct_outputs = target_distribution

            current_loss =  self.predict(self.weights)
            # after predict, we also have a new self.prediction_outputs array

            total_loss += current_loss

            self.weights_gradients =  compute_gradients(self.weights)
            self.bp_update_weights()
        return total_loss








"""

###Make neural network but change weigths manually, 
#then use Adam optimizer or something
import numpy as np
input_sample=np.array([1,2,3])

input_sample=torch.from_numpy(input_sample).long()
#input_sample=input_sample.long()
print(input_sample)

input_sample= torch.tensor(input_sample, dtype=torch.long)

print(type(input_sample).dtype)

model=Feedforward(3,2)
#for param in model.parameters():
#    print(param.data)

model.apply(weights_init_xavier)

for param in model.parameters():
    print(param.data)

out=model.forward(input_sample)

print(out)

"""
'''
Simple Multilayer Perceptron with Softmax and Cross-Entropy. Backpropagation handled using autograd package (automatic differentiation).
Author: Goran Trlin
Find more tutorials and code samples on:
https://playandlearntocode.com
'''
"""
import numpy
from PIL import Image
from classes.mlp.mlp import MLP
from classes.csv.csv_data_loader import CsvDataLoader
from classes.extraction.image_feature_extractor import ImageFeatureExtractor

print('MLP program starting...')

#Load training data from CSV files:
csv_data_loader = CsvDataLoader()
data_training =  csv_data_loader.get_training_data('./../csv/correct_distributions_for_training.txt')

mlps = [MLP(data_training)]

# TRAINING PHASE:

# BACKPROPAGATION SETTINGS:
TRAIN_ITERATIONS = 100
TARGET_ACCURACY = 2

for i in range(0, len(mlps)):
    total_loss = 99999
    total_delta = 9999
    train_count = 0

    while train_count < TRAIN_ITERATIONS and total_loss > TARGET_ACCURACY:
        total_loss = mlps[i].train_network()

        print('TOTAL LOSS AT ITERATION (' + str(train_count) + '):')
        print(total_loss)
        train_count += 1

    print ('Training stopped at step #' + str(train_count) + ' for i=' + str(i))

# TESTING PHASE:
test_file_name = input('enter the filename:')
test_file_path = './../test_images/' + test_file_name

image = Image.open(test_file_path)
image.show()

img_extractor = ImageFeatureExtractor()
(im_test_image, test_image_pixels) = img_extractor.load_image(test_file_path)
(test_image_f1, test_image_f2, test_image_f3) = img_extractor.extract_features(im_test_image, test_image_pixels)

test_image_row = [test_file_name, test_image_f1, test_image_f2, test_image_f3, 0,0]

print(test_image_row)
output_from_mlp = mlps[0].make_prediction(test_image_row)

print('Output:')
print(output_from_mlp)

labels = ['CIRCLE', 'LINE']

index =  numpy.argmax(output_from_mlp)
label = labels[index]

print ('This image is a ' + label)
print('MLP program completed.')

"""