#For data
import random
import math
import copy

#For Discord app
import numpy
import random as rand
import discord
import os
from dotenv import load_dotenv

TOKEN = 'OTE0OTQ4NzgwMDU3NTc5NTIw.YaUedA.Jc7SsmGPGmmxB4LS1_jRYEwcKyE' //this token has been reset
GUILD = 'Aed Dungeon'

client = discord.Client()
 
# This splits the dataset into folds
def make_folds(dataset, numFolds):
	dataset_split = []
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / numFolds)
	for i in range(numFolds):
		fold = []
		while len(fold) < fold_size:
			index = random.randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Initialize the feedforward network
def new_network(numInputs, numHidden, numOutputs):
	network = []
	hidden_layer = [{'weights':[random.random() for i in range(numInputs + 1)]} for i in range(numHidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random.random() for i in range(numHidden + 1)]} for i in range(numOutputs)]
	network.append(output_layer)
	return network
 
# Makes an x vector to be used
def find_x(weights, inputs):
	x = weights[-1]
	for i in range(len(weights)-1):
		x += weights[i] * inputs[i]
	return x

# This function forward propagates the input through the hidden layer until it reaches the output layer
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			x = find_x(neuron['weights'], inputs)
			neuron['output'] = 1.0 / (1.0 + math.exp(-x))
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
 
# This calculates and stores the errors as part of back propagation
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = []
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0
				for neuron in network[i + 1]:
					error = neuron['output'] * (1.0 - neuron['output']) #error: delta is always 0
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append((neuron['output'] - expected[j])*neuron['output']*(1-neuron['output'])) #expected is equal to output when it shouldnt be
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = 0.1 * errors[j] * neuron['output'] * (1.0 - neuron['output']) #delta is 0 because of errors being 0. You have to find out why errors are always 0.
 
# Updates network weights
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] -= l_rate * neuron['delta']
 
 
# Predicts Chad's mood in response to the input
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))
 
# This function follows a backpropagation algorithm 
def back_propagation(train, test, l_rate, iterations, numHidden):
	numInputs = len(train[0]) - 1
	numOutputs = len(set([row[-1] for row in train]))
	network = new_network(numInputs, numHidden, numOutputs)
	for iter in range(iterations):
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(numOutputs)]
			expected[row[-1]-1] = 1
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
			
	predictions = []
	for row in test:
		prediction = predict(network, row) + 1
		predictions.append(prediction)
	return predictions, network

# This generates the training data for the algorithm to run and tests how accurate it is. This is where most of the functions are assimilated.
def feedforward_network(dataset, numFolds, *args):
	folds = make_folds(dataset, numFolds)
	accuracies = []
	for fold in folds:
		training_data = list(folds)
		training_data.remove(fold)
		training_data = sum(training_data, [])
		testing_data = []
		for row in fold:
			row_copy = list(row)
			testing_data.append(row_copy)
			row_copy[-1] = None
		predicted, network = back_propagation(training_data, testing_data, *args)
		actual = [row[-1] for row in fold]
		correct = 0
		for i in range(len(actual)):
			if actual[i] == predicted[i]:
				correct += 1
		accuracy = correct / float(len(actual)) * 100.0
		accuracies.append(accuracy)
	return accuracies, network

random.seed(1)
# Reads in the dataset
filename = 'chimps.txt'
dataset = []
infile = open(filename, 'r')
for line in infile:
	row = []
	for i in line.strip().split(','):
		row.append(i)
	dataset.append(row)

# Saves labels
labels = copy.deepcopy(dataset[0])
dataset.pop(0)

#Encodes boolean values of dataset into integers
for i in range(len(dataset[0]) - 1):
	for row in dataset:
		if row[i] == 'TRUE':
			row[i] = '1'
		elif row[i] == 'FALSE':
			row[i] = '0'
		row[i] = float(row[i].strip())

# Parsing the class column into ints so that we can use them as indices
for row in dataset:
	row[len(dataset[0]) - 1] = int(row[len(dataset[0]) - 1].strip())

numFolds = 5
learning_rate = 0.1
numIter = 1000
numHidden = 5
accuracies, network = feedforward_network(dataset, numFolds, learning_rate, numIter, numHidden)
print("Accuracies: ")
for i in accuracies:
	print(i)

# This lets Chad get online
@client.event
async def on_ready():
    for guild in client.guilds:
        if guild.name == GUILD:
            break

    print(f'{client.user} is connected to the following guild:\n'f'{guild.name}(id: {guild.id})')

# When someone new joins a server
@client.event
async def on_member_join(member):
    await member.create_dm()
    await member.dm_channel.send(f'Ook ook eek! (Hi {member.name}, welcome to the jungle!)')

# When a message is sent to Chad's server or his direct messages
@client.event
async def on_message(message):
    if message.author == client.user:
        return
    bored_chad = [
        '*Chad stares into nothingness*',
        '*Chad is loafing around*',
        '*Belches*',
        '...',
        '*yawns*'
    ]

    active_chad = [
        'Ook?',
        '*Chad stares at you*',
        'Eek ook *Chad throws a stick*'
    ]

    enraged_chad = [
        'REEEEEEE!',
        '*Chad bares his fangs*',
        '*Chad slams his hands on the ground. You\'ve somehow offended him!*'
    ]

    if 'chad' in message.content.lower():
        v1 = rand.uniform(1,12)
        v2 = rand.uniform(1,4)
        input = [1, 1, 1, v1, v2, None]
        mood = predict(network, input) + 1
        if mood == 1:
            response = rand.choice(bored_chad)
        elif mood < 4:
            response = rand.choice(active_chad)
        elif mood >= 4:
            response = rand.choice(enraged_chad)
        print(mood, v1, v2)
        await message.channel.send(response)

    elif 'stinks' in message.content.lower():
        v1 = rand.uniform(10,12)
        v2 = rand.uniform(1,4)
        input = [0, 0, 1, v1, v2, None]
        mood = predict(network, input) + 1
        if mood == 1:
            response = rand.choice(bored_chad)
        elif mood < 4:
            response = rand.choice(active_chad)
        elif mood >= 4:
            response = rand.choice(enraged_chad)
        print(mood, v1, v2)
        await message.channel.send(response)



client.run(TOKEN)
