# HAR classification 
# Author: Burak Himmetoglu
# 8/15/2017

import pandas as pd 
import numpy as np
import os

def read_data(data_path, split = "train"):
	""" Read data """

	# Fixed params
	n_class = 5
	n_steps = 3000

	# Paths
	path_ = os.path.join(data_path, split)
	path_signals = os.path.join(path_, "raw")

	# Read labels and one-hot encode
	label_path = os.path.join(path_, "y_train.csv")
	labels = pd.read_csv(label_path, header = None)

	# Read time-series data

	# Initiate array
	X = np.zeros((len(labels), n_steps,1))
	dat_ = pd.read_csv(os.path.join(path_signals,"xb_res.txt"), sep = ',', delim_whitespace = False, header = None)
	Raw = np.zeros((len(dat_.as_matrix()),n_steps))
	X[:,:,0]= dat_.as_matrix()

	# Return 
	return X, labels[0].values

def standardize(train, test):
	""" Standardize data """
	all_data = np.concatenate((train,test), axis = 0)
	assert np.allclose(all_data[:len(train)],train), "Wrong training set!"
	assert np.allclose(all_data[len(train):], test), "Wrong test set!"

	# Standardise each channel
	all_data = (all_data - np.mean(all_data, axis=1)[:,None]) / np.std(all_data, axis=1)[:,None]

	# Split back and return
	X_train = all_data[:len(train)]
	X_test = all_data[len(train):]

	return X_train, X_test

def one_hot(labels, n_class = 6):
	""" One-hot encoding """
	expansion = np.eye(n_class)
	y = expansion[:, labels-1].T
	assert y.shape[1] == n_class, "Wrong number of labels!"

	return y

def get_batches(X, y, batch_size = 100):
	""" Return a generator for batches """
	n_batches = len(X) // batch_size
	X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

	# Loop over batches and yield
	for b in range(0, len(X), batch_size):
		yield X[b:b+batch_size], y[b:b+batch_size]
	




