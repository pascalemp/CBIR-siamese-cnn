import sys
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

import cv2
import time

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.initializers import RandomNormal

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import initializers
from keras import backend as K


from sklearn.utils import shuffle

import numpy.random as rng



def get_siamese_model(input_shape):

    
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    
    model = Sequential()
    model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None), kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu',
                     kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                     bias_initializer=RandomNormal(mean=0.5, stddev=0.01, seed=None), kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                     bias_initializer=RandomNormal(mean=0.5, stddev=0.01, seed=None), kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                     bias_initializer=RandomNormal(mean=0.5, stddev=0.01, seed=None), kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),bias_initializer=RandomNormal(mean=0.5, stddev=0.01, seed=None)))
    
    # encoded feature vectors
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # lambda layer for l1 distance 
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # dense layer with sigmoid
    prediction = Dense(1,activation='sigmoid',bias_initializer=RandomNormal(mean=0.5, stddev=0.01, seed=None))(L1_distance)

    # create model
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return model
    return siamese_net

model = get_siamese_model((150, 150, 1)) #change to w,h,1 of image dimension if required.
model.summary()

optimizer = Adam(lr = 0.00006)
model.compile(loss="binary_crossentropy",optimizer=optimizer)

with open('train.pickle', "rb") as f: # Train data in pickle file after running dataset setup file.
    (Xtrain, train_classes) = pickle.load(f)
    
print("Training alphabets: \n")
print(list(train_classes.keys()))

with open('val.pickle', "rb") as f:
    (Xval, val_classes) = pickle.load(f)

print("Validation alphabets:", end="\n\n")
print(list(val_classes.keys()))

def get_batch(batch_size,s="train"):
    """Create batch of n pairs, half same class, half different class"""

    if s == 'train':
        X = Xtrain # X training input
        categories = train_classes # y categories
    else:
        X = Xval # X validation input
        categories = val_classes # y categories

    n_classes, n_examples, w, h = X.shape[0], X.shape[1], X.shape[2], X.shape[3]

    # randomly sample several classes to use in the batch of size n
    categories = rng.choice(n_classes,size=(batch_size,),replace=False)
    
    # initialize 2 empty arrays for the input image batch
    pairs=[np.zeros((batch_size, h, w,1)) for i in range(2)]
    
    # initialize vector for the targets
    targets=np.zeros((batch_size,))
    
    # one half of is full of '1's and 2nd half of batch has same class

    targets[batch_size//2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = rng.randint(0, n_examples)
        pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, 1)
        idx_2 = rng.randint(0, n_examples)
        
        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category  
        else: 
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1,n_classes)) % n_classes
        
        pairs[1][i,:,:,:] = X[category_2,idx_2].reshape(w, h,1)

    
    return pairs, targets

def generate(batch_size, s="train"):
    # Generator for batches to be used by network.
    while True:
        pairs, targets = get_batch(batch_size,s)
        yield (pairs, targets)

def make_oneshot_task(N, s="val", language=None):
    # Create pairs of images for support set to test K-One Shot Learning
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    
    indices = rng.randint(0, n_examples,size=(N,))
    if language is not None: # if language (category) is specified, select items in that category
        low, high = categories[language]
        if N > high - low:
            raise ValueError("This language ({}) has less than {} letters".format(language, N))
        categories = rng.choice(range(low,high),size=(N,),replace=False)

    else: # if no language (category) specified just pick random items from categories
        categories = rng.choice(range(n_classes),size=(N,),replace=False)            
    true_category = categories[0]
    ex1, ex2 = rng.choice(n_examples,replace=False,size=(2,))
    test_image = np.asarray([X[true_category,ex1,:,:]]*N).reshape(N, w, h,1)
    support_set = X[categories,indices,:,:]
    support_set[0,:,:] = X[true_category,ex2]
    support_set = support_set.reshape(N, w, h,1)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(targets, test_image, support_set)
    pairs = [test_image,support_set]

    return pairs, targets

def test_oneshot(model, N, k, s = "val", verbose = 0):
    # Testing average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
    for i in range(k):
        inputs, targets = make_oneshot_task(N,s)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct+=1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Acquired average of {}% {} way one-shot learning performance \n".format(percent_correct,N))
    return percent_correct

# Hyper parameters
evaluate_every = 500 # interval for evaluating on one-shot tasks
batch_size = 32
n_iter = 20000 # No. of training iterations
N_way = 10 # how many classes for testing one-shot tasks
n_val = 5 # how many one-shot tasks to validate on (this is the 'k' val in performance)
best = -1

model_path = './weights/' # Save our weights

print("Starting training process!")
print("-------------------------------------")
print('X Shape: ' + str(Xtrain.shape))
print("-------------------------------------")
t_start = time.time()
for i in range(1, n_iter+1):
    (inputs,targets) = get_batch(batch_size)
    loss = model.train_on_batch(inputs, targets)
    if i % evaluate_every == 0:
        print("\n ------------- \n")
        print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
        print("Train Loss: {0}".format(loss)) 
        val_acc = test_oneshot(model, N_way, n_val, verbose=True)
        model.save_weights(os.path.join('weights/', 'weights.{}.h5'.format(i)))
        if val_acc >= best:
            print("Current best: {0}, previous best: {1}".format(val_acc, best))
            best = val_acc

'''

Code based around implementation by https://github.com/kevinzakka/one-shot-siamese

'''
