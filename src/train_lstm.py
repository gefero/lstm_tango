#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 15:32:23 2021

@author: grosati
"""
#%% load packages
import numpy as np
import os
import random
import sys
import io
#%% load and preprocess data

with io.open('./data/V2letras.txt', encoding="utf-8") as f:
    text = f.read().lower()

text = text.replace("\n", " ")  # We remove newlines chars for nicer display
print("Corpus length:", len(text))

chars = sorted(list(set(text)))
print("Total chars:", len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print("Number of sequences:", len(sentences))

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
    
#%% load keras

#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, TensorBoard
    
#%% build lstm

print('Build model...')
model = keras.Sequential()

model.add(layers.LSTM(128, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(layers.Dropout(0.5))
model.add(layers.LSTM(128))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(chars), activation='softmax'))


optimizer = keras.optimizers.RMSprop(learning_rate=0.005)
model.compile(loss="categorical_crossentropy", 
              metrics=['accuracy'],
              optimizer=optimizer)

model.summary()
#%% defining callbacks

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)


checkpoint_path = "./checkpoints/lstm_layers2_batch64.{epoch:02d}.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = ModelCheckpoint(checkpoint_path, 
                              save_weights_only=False,
                              save_best_only=True, 
                              monitor='loss',
                              mode='min',
                              verbose=1)


tbCallBack = TensorBoard(log_dir='./log', 
                         histogram_freq=0,
                         write_graph=False,
                         write_grads=True,
                         batch_size=10,
                         update_freq='batch',
                         write_images=True)

#%%

model.fit(x, y,
          batch_size=64,
          epochs=50,
          callbacks=[cp_callback, tbCallBack, print_callback])

#%%

print('Saving full model...')
model.save('models/lstm_2_64_M')
print('Done')
