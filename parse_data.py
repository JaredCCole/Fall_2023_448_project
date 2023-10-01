import spacy
import multiprocess as mp
import time


nlp = spacy.load("en_core_web_sm")


'''
	This file is responsible for parsing the data into a (word, label) pair
	format. It also extracts features and converts the data into a more usable 
	feature_vector representation.

	TO DO:
	* format data into (word, label) pairs      (Done)
		* remove blank data                     (Done)
		* remove nonsensical data               (In Progress)
	* Define Basic Features                     (Done)
	* Split the data into training and dev-test (Not started)
	* Extract Features from each input          (Done)
	* Parallelize feature extraction            (Done)

'''


def parse(path: str) -> list:
	'''
	Input: path to the training data
	Output: [(word, label),(word, label),...,(label)]
	'''

	#read the input file
	with open(path, "r", encoding='utf-8') as f:
		contents = f.read()

	contents = contents.split("\n")

	# remove empty strings from the list
	contents = [i for i in contents if i != ""]

	set_labels = set()

	# (word, label) pairs
	pairs = [(i.split(" ")[0], i.split(" ")[1]) for i in contents]


	for i in contents:
		set_labels.add(i[1])

	return pairs





'''
Notes:
 	Basic Feature Definitions:
	- Word form
	- Suffix
	- Prefix
	- Capitalization
	- Word lemma (base form)
	- Word Frequency
	- word length
	- ends in "ly" --> most likely an adverb
	- "a", "an", "the" --> DT
	- ends in "ed" --> most likely past tense
	- "and", "but", and "or" --> CC

	
Feature structure

	example: 'The'

	{
		'word_form': 'The',  # the word itself
		'prefix': 'Th'       # prefix of the word
		'suffix': 'he'       # suffix of the word
		'capitalized': 1,    # can be a 0 or 1
		'base_form': 'the'   # base form of the word (use spacy to find this)
		'frequency':         #(occurences of 'The')/(total_words_in_train_set) ?? I'm not sure about this one
		'word_length': 3,
		'ends_in_ly': 0,     # can be a 0 or 1
		'a_an_the': 1,       # can be a 0 or 1
		'ends_in_ed': 0,     # can be a 0 or 1
		'ends_in_ing': 0,    # can be a 0 or 1
		'':
		...

	}

'''

# helper function for creating features

def construct_feature(input_word_pair) -> dict:
	'''
	Input:	input_word is the word we are constructing features for
	Output: feature
	'''
	input_word = input_word_pair[0]

	feature = dict()
	feature['word_form'] = input_word
	feature['prefix'] = input_word[:2]
	feature['suffix'] = input_word[-2:]
	feature['capitalized'] = input_word.istitle()
	#feature['base_form'] = nlp(input_word)[0].lemma_
	feature['word_length'] = input_word
	feature['ends_in_ly'] = 1 if input_word[-2:] == 'ly' else 0
	feature['ends_in_ed'] = 1 if input_word[-2:] == 'ed' else 0
	feature['ends_in_ing'] = 1 if input_word[-3:] == 'ing' else 0
	feature['a_an_the'] = 1 if input_word in ['a', 'an', 'the'] else 0

	return feature


# Feature extraction

def extract_features(data: list) -> list:
	'''
	Input: 
	Output: list of dictionaries
		* each dictionary is a feature

	Feature structure looks like this
	e.g.
		x0 = {'word_form': 'The',
			  'prefix': Th,
			  'suffix': he,
			  'capitalized': 1,
			  ...

			  'conjucntion': 0}
	'''

	start = time.time()

	features = []

	# features = [construct_feature(word) for (word, label) in data]

	with mp.Pool() as pool:
		features = pool.map(construct_feature, data)


	end = time.time()

	print(f"* Feature extraction took: {end - start}")

	return features







