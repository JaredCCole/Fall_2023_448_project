import torch
import json

'''
	This file contains the code for a bayesian classifier written in pytorch. This
	file loads the json file "./features.json" that contains the input data as a 
	list of dictionaries representing the input features for each data point. 

	TO DO:
		* Convert data into tensors                     (In Progress)
			* need to represent the words numerically   (Not Done)
				--> i am thinking we should use one hot 
					encoding 
		* Build the model                               (In Progress)
			* 
		* break the data into batches

'''

def build_tensors(features):

	feature_tensors = {
		#'word_form': torch.tensor([int(d['word_form']) for d in features]),
		#'prefix': torch.tensor([int(d['prefix']) for d in features]),
		#'suffix': torch.tensor([int(d['suffix']) for d in features]),
		'capitalized': torch.tensor([d['capitalized'] for d in features], dtype=torch.bool),
		'word_length': torch.tensor([d['word_length'] for d in features], dtype=torch.long),
		'ends_in_ly': torch.tensor([d['ends_in_ly'] for d in features], dtype=torch.bool),
		'ends_in_ed': torch.tensor([d['ends_in_ed'] for d in features], dtype=torch.bool),
		'ends_in_ing': torch.tensor([d['ends_in_ing'] for d in features], dtype=torch.bool),
		'a_an_the': torch.tensor([d['a_an_the'] for d in features], dtype=torch.bool),
	}

	return feature_tensors






if __name__ == "__main__":

	# first load the "./features.json" file
	data = json.load(open('./features.json'))

	# convert the data into tensors
	feature_tensors = build_tensors(data)

	# instantiate the model




