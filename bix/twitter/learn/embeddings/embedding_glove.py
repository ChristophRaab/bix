import os
from typing import List

import tensorflow
import urllib.request
import zipfile
from keras import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras_preprocessing.text import Tokenizer
from numpy import asarray, zeros
from nltk import SnowballStemmer

from bix.twitter.learn.embeddings.embedding_abstract import EmbeddingAbstract


class EmbeddingGlove(EmbeddingAbstract):
    def __init__(self, tokenizer: Tokenizer, padded_x, unpadded_x, max_tweet_word_count: int, vocab_size: int, y: List[int]) -> None:
        super().__init__(tokenizer, padded_x, unpadded_x, max_tweet_word_count, vocab_size, y)

        self.embedding_vector_size = 100
        self.embedding_matrix = None
        self.weights = None

    def prepare(self):
        # load the whole embedding into memory
        embeddings_index = dict()

        if not os.path.isfile('glove.twitter.27B.100d.txt'):
            print('file: glove.twitter.27B.100d.txt does not exist. Downloading it...')
            urllib.request.urlretrieve('http://nlp.stanford.edu/data/glove.twitter.27B.zip', 'glove.twitter.27B.zip')
            print('finished downloading')
            print('unpacking...')
            with zipfile.ZipFile('glove.twitter.27B.zip', 'r') as zip_ref:
                zip_ref.extractall('.')
            print('finished unpacking')

        f = open('glove.twitter.27B.100d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))

        # create a weight matrix for words in training docs
        embedding_matrix = zeros((self.vocab_size, self.embedding_vector_size))
        print(str(embedding_matrix.shape))
        stemmer = SnowballStemmer('english')
        for word, i in self.tokenizer.word_index.items():
            if i == self.vocab_size: break
            word_stemmed = stemmer.stem(word)
            embedding_vector = embeddings_index.get(word_stemmed)  # todo: stemm glove index
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        self.embedding_matrix = embedding_matrix

    def define_model(self):
        # define model
        model = Sequential()
        e = Embedding(self.vocab_size, self.embedding_vector_size, weights=[self.embedding_matrix],
                      input_length=self.max_tweet_word_count, trainable=False)
        model.add(e)
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))  # verstehen und vieleicht für ba relevant
        # compile the model

        #run_opts = tensorflow.RunOptions(report_tensor_allocations_upon_oom=True)
        # compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])#, options=run_opts)
        # summarize the model
        print(model.summary())

        self.model = model

    def learn(self):
        # fit the model
        #self.model.fit(self.x, self.y, epochs=5, verbose=1) # todo: increase epocs
        # evaluate the model
        #loss, accuracy = self.model.evaluate(self.x, self.y, verbose=0)
        #print('Accuracy: %f' % (accuracy * 100))
        print("no learning needed (pretrained embedding)")

        weights = self.model.layers[0].get_weights()
        print(f"num: {len(weights)}, dim: {weights[0].shape}")

        self.weights = weights

    def get_weights(self):
        return self.weights

