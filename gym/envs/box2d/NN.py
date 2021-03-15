# code modified from: Jason Brownlee on November 7, 2016 in Code Algorithms From Scratch, https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)
import numpy as np
from math import exp
from random import seed
from random import random
class NeuralNetwork():
    # Initialize a network
    def __init__(self, n_inputs, hidden, n_outputs):
        self.network = list()
        self.fitness = 0
        self.size = [n_inputs]
        for i in hidden:
            self.size.append(i)
        self.size.append(n_outputs)
        prev_nodes = n_inputs
        for n_hidden in hidden:
            hidden_layer = [{'weights':[random() for i in range(prev_nodes + 1)]} for i in range(n_hidden)] #last weight is bias
            self.network.append(hidden_layer)
            prev_nodes = n_hidden
        output_layer = [{'weights':[random() for i in range(prev_nodes + 1)]} for i in range(n_outputs)]
        self.network.append(output_layer)

    # Calculate neuron activation for an input
    def activate(self, weights, inputs):
        activation = weights[-1] #bias
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation

    # Transfer neuron activation
    def sigmoid(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    # Forward propagate input to a network output
    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.sigmoid(activation)
                new_inputs.append(neuron['output'])
            inputs = 2*np.array(new_inputs) - 1 #map to -1 to 1
        return inputs


    def mutate(self, MutationChance=0.5, MutationStrength=0.5):
        for layer in self.network:
            for neuron in layer:
                for i in range(len(neuron['weights'])):
                    if random() < MutationChance:
                        change = random()*2 - 1
                        neuron['weights'][i] += MutationStrength*change

    def keras_weightswap(self, weights):
        prev_nodes = self.size[0]
        for layer, nodes in enumerate(self.size[1:]):
            for node in range(nodes):
                for w in range(prev_nodes - 1):
                    self.network[layer][node]['weights'][w] = weights[2*layer][:, node][w]
                self.network[layer][node]['weights'][-1] = weights[2*layer + 1][node] #bias
            prev_nodes = nodes
def create_training_set(setSize, trainer):
    train_x = []
    train_y = []
    for i in range(setSize):
        train_x.append(2*np.random.rand(trainer.size[0])-1)
        train_x[i][0] = abs(train_x[i][0]) #speed input
        train_y.append(trainer.forward_propagate(train_x[i]))
    return np.array(train_x), np.array(train_y)


'''
from tensorflow.keras import Sequential, layers, losses
model = Sequential()

model.add(layers.Dense(8, input_dim=4, activation='sigmoid'))
model.add(layers.Dense(8, activation='sigmoid'))
model.add(layers.Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
NN = NeuralNetwork(4, [8, 8], 2)
weights = model.get_weights()
print(weights)
NN.copy_keras_weights(weights)
'''