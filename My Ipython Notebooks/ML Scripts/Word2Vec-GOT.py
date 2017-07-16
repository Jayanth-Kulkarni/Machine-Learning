# create word vectors from GOT and analyze
# them semantically

from __future__ import absolute_import,division,print_function
# for word encoding
import codecs
# regex
import glob
'''
concurrency
for running parallel threads
'''
import multiprocessing
# for reading files etc
import os
# pretty printing
import pprint
# regular expressing
import re
# natural language toolkit
import nltk
# word 2 vector
import gensim.models.word2vec as w2v
# dimentionality reduction
import sklearn.manifold
# math
import numpy as np
# plotting
import matplotlib.pyplot as plt
# parse pandas
import pandas as pd
# visualization
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')

# get the book name
book_filenames = sorted(glob.glob("/home/jayanth/Work/Siraj Raval/data/*.txt"))
print(book_filenames)

corpus_raw = u""
for book_filename in book_filenames:
    print("Reading '{0}'....".format(book_filename))
    with codecs.open(book_filename,"r","utf-8") as book_file:
        corpus_raw += book_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()

tokenizer = nltk.data.load('/home/jayanth/nltk_data/tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)
# convert into a list of words
# remove unwanted characters
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]","",raw)
    words = clean.split
    return words

my_sentences = [ ]
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        my_sentences.append(sentence_to_wordlist(raw_sentence))

print(raw_sentences[1])
print(sentence_to_wordlist(raw_sentences[5]))

# Train word2vec

'''
Vectors help with 3 main tasks
distances, similarity and ranking
'''

# dimentionality of the word vectors.

num_features = 300
# vectors are type of tensors
# minimum word count threshold
min_word_count = 3

# Number of threads that run in parallel
num_workers = multiprocessing.cpu_count()

# context window length
context_size = 7

# downsample setting for frequent words
# how often we need to look at the words
downsample = 1e-3

# seed for the RNG
seed = 1

GOT2vec = w2v.Word2Vec(
    sg = 1,
    seed = seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsample
)

GOT2vec.build_vocab(sentences=my_sentences)

print("word2vec vocab length:",len(GOT2vec.vocab))

GOT2vec.train(my_sentences)
