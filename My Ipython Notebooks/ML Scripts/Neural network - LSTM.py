# Long Short Term Memory to generate cities
# to avoid version errors
from __future__ import absolute_import, division, print_function

import os
# pull data from internet
from six import moves
import ssl
import tflearn
from tflearn.data_utils import *

# Retrieving the data

path = "US_cities.txt"
if not os.path.isfile(path):
    context = ssl._create_unverified_context()
    # get data text
    moves.urllib.request.urlretrieve("https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/US_Cities.txt", path, context=context)

# city name max length
maxlen = 20

# vectorize the text file
# X is input Y is target char_idx is dictionary
X, Y, char_idx = textfile_to_semi_redundant_sequences(path,
    seq_maxlen=maxlen,redun_step=3)

# Create LSTM
g = tflearn.input_data(shape=[None,maxlen,len(char_idx)])
# 512 neurons
g = tflearn.lstm(g,512,return_seq=True)
# LSTM is a subset of neural network
# dropout takes out some neural pathways so as to avoid overfitting
# it randomly turns it off
g = tflearn.dropout(g,0.5)
g = tflearn.lstm(g,512)
g = tflearn.dropout(g,0.5)
# softmax is a type of logistic regression that is good for classification
g = tflearn.fully_connected(g,len(char_idx),activation='softmax')
#
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.001)
# generate cities
# checkpoint_path saves the checkpoints as we train
m = tflearn.SequenceGenerator(g,dictionary=char_idx,
                seq_maxlen=maxlen,
                clip_gradients=5.0,
                checkpoint_path='model_us_cities')

# training
for i in range(40):
    # seed helps us start from the same point when we generate new things
    # as we want our name to be similar to the ones in the data file
    seed = random_sequence_from_textfile(path,maxlen)
    '''
    takes the model - from SequenceGenerator takes the
    training data and adds it to our LSTM
    '''
    m.fit(X,Y,validation_set=0.1,batch_size=128,
    n_epoch=1,run_id='us_cities')
    print("-- Test with temperature of 1.2 --")
    print(m.generate(30,temperature=1.2,seq_seed=seed))
    print("-- Test with temperature of 1.0 --")
    print(m.generate(30,temperature=1.0,seq_seed=seed))
    print("-- Test with temperature of 0.5 --")
    print(m.generate(30,temperature=0.5,seq_seed=seed))
