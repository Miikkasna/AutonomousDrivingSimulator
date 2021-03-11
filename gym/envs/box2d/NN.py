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
        #print(self.size)

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

    # Calculate the derivative of an neuron output
    def sigmoid_derivative(self, output):
        return output * (1.0 - output)

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.sigmoid_derivative(neuron['output'])

    # Update network weights with error
    def update_weights(self, row, l_rate):
        for i in range(len(self.network)):
            inputs = row
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']
    def mutate(self, MutationChance=0.5, MutationStrength=0.5):
        for layer in self.network:
            for neuron in layer:
                for i in range(len(neuron['weights'])):
                    if random() < MutationChance:
                        change = random()*2 - 1
                        neuron['weights'][i] += MutationStrength*change
    # Train a network for a fixed number of epochs
    def train_network(self, train_x, train_y, l_rate, n_epoch):
        for epoch in range(n_epoch):
            sum_error = 0
            for i, row in enumerate(train_x):
                outputs = self.forward_propagate(row)
                expected = train_y[i]
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(row, l_rate)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
            #l_rate -= 0.003
    
'''
# Test training backprop algorithm
seed(1)
dataset_x = [[2.7810836,2.550537003],
	[1.465489372,2.362125076],
	[3.396561688,4.400293529],
	[1.38807019,1.850220317],
	[3.06407232,3.005305973],
	[7.627531214,2.759262235],
	[5.332441248,2.088626775],
	[6.922596716,1.77106367],
	[8.675418651,-0.242068655],
	[7.673756466,3.508563011]]
dataset_y = [[0,1],
	[0,1],
	[0,1],
	[0,1],
	[0,1],
	[0,1],
	[0,1],
	[0,1],
	[0,1],
	[0,1]]
n_inputs = len(dataset_x[0])
n_outputs = len(dataset_y[0])
print(n_outputs)
NN = NeuralNetwork(n_inputs, [3, 3], n_outputs)
NN.train_network(dataset_x, dataset_y, 0.1, 20)
for layer in NN.network:
	print(layer)
print(NN.forward_propagate([0.2, 0.3]))
'''