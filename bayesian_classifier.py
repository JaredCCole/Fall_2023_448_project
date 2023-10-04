import torch
from torch.utils.data import TensorDataset, DataLoader
import json
import time

'''
	This file contains the code for a bayesian classifier written in pytorch. This
	file loads the json file "./features.json" that contains the input data as a 
	list of dictionaries representing the input features for each data point. 

	TO DO:
		* Convert data into tensors                     (In Progress)
			* need to represent the words numerically   (Done)
			* convert the labels into a tensor          (In Progress)
		* Build the model                               (In Progress)
			* 
		* have a set of labels and map them to numbers

'''

# Each word should be mapped to an integer
# Global variable (vocab)
vocab = {}

def construct_vocab(features):
	for n,d in enumerate(features):
		vocab[d['word_form']] = n




def build_tensors(features):

	feature_tensors = {
		'word_form': torch.tensor([vocab[d['word_form']] for d in features]),
		'capitalized': torch.tensor([d['capitalized'] for d in features], dtype=torch.bool),
		'word_length': torch.tensor([d['word_length'] for d in features], dtype=torch.long),
		'suffix': torch.tensor([d['suffix'] for d in features], dtype=torch.long),
		'a_an_the': torch.tensor([d['a_an_the'] for d in features], dtype=torch.bool),
		'and_or_but': torch.tensor([d['and_or_but'] for d in features], dtype=torch.bool),
		'comma': torch.tensor([d['comma'] for d in features], dtype=torch.bool),
		'period': torch.tensor([d['period'] for d in features], dtype=torch.bool),
		'dollar_sign': torch.tensor([d['dollar_sign'] for d in features], dtype=torch.bool),
		'single_quotes': torch.tensor([d['single_quotes'] for d in features], dtype=torch.bool),
		'contains_number': torch.tensor([d['contains_number'] for d in features], dtype=torch.bool),
		'plus_or_equals': torch.tensor([d['plus_or_equals'] for d in features], dtype=torch.bool),
		'prefix': torch.tensor([d['prefix'] for d in features], dtype=torch.long),
	}

	return feature_tensors





'''
class BayesianClassifier:

	def __init__(self):
		# init the model here
		


	def train(self, X, Y, epochs):
		'''
		Input:
			X: tensor containing features
			Y: tensor containing labels
		Output:
			model parameters
		'''
		return


	def predict(self, X):
		'''
		Input:
			X: tensor containing input features
		Output:
			Y: tensor containing predicted labels
		'''

		return
'''



if __name__ == "__main__":
	start = time.time()
	# first load the "./features.json" file
	data = json.load(open('./features.json'))

	# construct the vocabulary
	construct_vocab(data)


	# convert the data into tensors
	feature_tensors = build_tensors(data)

	end = time.time()
	print(f"* Building tensors took: {end - start}")


	# convert labels into tensors
	labels = []


	# dataset
	#dataset = TensorDataset(feature_tensors, labels)


	# instantiate the model




