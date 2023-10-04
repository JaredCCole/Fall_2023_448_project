import torch
import json
import time

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
		'ends_in_ly': torch.tensor([d['ends_in_ly'] for d in features], dtype=torch.bool),
		'ends_in_ed': torch.tensor([d['ends_in_ed'] for d in features], dtype=torch.bool),
		'ends_in_ing': torch.tensor([d['ends_in_ing'] for d in features], dtype=torch.bool),
		'a_an_the': torch.tensor([d['a_an_the'] for d in features], dtype=torch.bool),
		'ends_in_tion_sion': torch.tensor([d['ends_in_tion_sion'] for d in features], dtype=torch.bool),
		'ends_in_ment': torch.tensor([d['ends_in_ment'] for d in features], dtype=torch.bool),
		'ends_in_ies': torch.tensor([d['ends_in_ies'] for d in features], dtype=torch.bool),
		'ends_in_xes': torch.tensor([d['ends_in_xes'] for d in features], dtype=torch.bool),
		'ends_in_able_ible': torch.tensor([d['ends_in_able_ible'] for d in features], dtype=torch.bool),
		'ends_in_ence': torch.tensor([d['ends_in_ence'] for d in features], dtype=torch.bool),
		'ends_in_er_or': torch.tensor([d['ends_in_er_or'] for d in features], dtype=torch.bool),
		'and_or_but': torch.tensor([d['and_or_but'] for d in features], dtype=torch.bool),
		'comma': torch.tensor([d['comma'] for d in features], dtype=torch.bool),
		'period': torch.tensor([d['period'] for d in features], dtype=torch.bool),
		'dollar_sign': torch.tensor([d['dollar_sign'] for d in features], dtype=torch.bool),
		'single_quotes': torch.tensor([d['single_quotes'] for d in features], dtype=torch.bool),
		'contains_number': torch.tensor([d['contains_number'] for d in features], dtype=torch.bool),
		'plus_or_equals': torch.tensor([d['plus_or_equals'] for d in features], dtype=torch.bool),
		'begins_with_un': torch.tensor([d['begins_with_un'] for d in features], dtype=torch.bool),
		'begins_with_in_il_im': torch.tensor([d['begins_with_in_il_im'] for d in features], dtype=torch.bool),
		'begins_with_dis': torch.tensor([d['begins_with_dis'] for d in features], dtype=torch.bool),
		'begins_with_re': torch.tensor([d['begins_with_re'] for d in features], dtype=torch.bool),
	}

	return feature_tensors






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


	# instantiate the model




