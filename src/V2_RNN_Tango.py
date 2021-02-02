#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install h5py pyyaml ')
get_ipython().system('pip install -U -q PyDrive')

get_ipython().system('mkdir ./chkpt/')
get_ipython().system('mkdir ./log/')


# In[2]:


get_ipython().system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
get_ipython().system('unzip ngrok-stable-linux-amd64.zip')
LOG_DIR = './log'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)

get_ipython().system_raw('./ngrok http 6006 &')

get_ipython().system('curl -s http://localhost:4040/api/tunnels | python3 -c     "import sys, json; print(json.load(sys.stdin)[\'tunnels\'][0][\'public_url\'])"')


# In[3]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[1]:


import numpy as np
import os
import random
import sys
import io


# In[5]:


#path = './gdrive/My Drive/Notebooks/RNN Tango/data/V2letras.txt'
path = '../data/V2letras.txt'


with open(path, encoding='utf-8') as f:
    text_orig = f.read().lower()

#text = open('./gdrive/My\ Drive/Notebooks/RNN\ Tango/data') 


# In[6]:


#text = str(uploaded.values())
#text = text_orig[0:1005000]
text = text_orig


# In[7]:


from __future__ import print_function
from keras.callbacks import LambdaCallback, ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file


# In[8]:


print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# In[9]:


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(lr=0.005)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)


# In[10]:


model.summary()


# In[11]:


# build callbacks

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
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


checkpoint_path = "./gdrive/My Drive/Notebooks/RNN Tango/chkp/lstm_2_128_M.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = ModelCheckpoint(checkpoint_path, 
                              save_weights_only=False,
                              save_best_only=True, 
                              monitor='loss',
                              mode='min',
                              verbose=1)


from keras.callbacks import TensorBoard
tbCallBack = TensorBoard(log_dir='./log', 
                         histogram_freq=0,
                         write_graph=False,
                         write_grads=True,
                         batch_size=10,
                         update_freq='batch',
                         write_images=True)


# In[18]:


model.fit(x, y,
          batch_size=128,
          epochs=60,
          callbacks=[cp_callback, tbCallBack, print_callback])


# In[ ]:





# In[ ]:


# load the model
from keras.models import load_model
from numpy.testing import assert_allclose

new_model = load_model(checkpoint_path)
assert_allclose(model.predict(x),
                new_model.predict(x),
                1e-5)


# In[ ]:


model_path = './gdrive/My Drive/Notebooks/RNN Tango/model/lstm_2_128.hdf5'
model.save(model_path)


# In[15]:


text


# In[19]:


model.evaluate(x,y)


# In[ ]:




