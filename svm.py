import torch
import json

'''
	This file contains the code for support vector machines written in pytorch. This
	file loads the json file "./features.json" that contains the input data as a 
	list of dictionaries representing the input features for each data point. 
'''

def build_tensors(features):
	
    # TODO: Same for each algorithm? 

    return feature_tensors

def run_svm_model(feature_tensors):
	
	# TODO: Create svm model

    return

if __name__ == "__main__":

	# Load the "./features.json" file and covert to tensors
    data = json.load(open('./features.json'))
    feature_tensors = build_tensors(data)

    #Run svm model
    run_svm_model(feature_tensors)