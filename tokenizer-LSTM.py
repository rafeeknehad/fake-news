import itertools
import cnn
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import models,layers
# Importing libraries
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import  AveragePooling1D
from sklearn.metrics import accuracy_score, classification_report
from keras.preprocessing import sequence
from sklearn import metrics
from tensorflow.python.ops.confusion_matrix import confusion_matrix
from tensorflow.keras.preprocessing.text import one_hot
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#import emoji

df=pd.read_csv('covid_clean_dataset_09_02_21 (1) (1).csv')

    
#df["text"] = df["text"].apply(remove_contractions)
#df["text"] = df["text"].apply(clean_text)
#df.drop_duplicates(subset=["text"], inplace=True)
#df.dropna(inplace=True)
print(df.head)
voc_size=5000
sent_length=100
X_train, X_test, y_train, y_test = train_test_split(df.text, df.target, test_size=0.3, random_state=37)
tk = Tokenizer(num_words=10000,
                  filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n', lower=True, split=" ")
tk.fit_on_texts(X_train)
X_train_seq = tk.texts_to_sequences(X_train)
X_test_seq = tk.texts_to_sequences(X_test)

X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=100)
X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=100)
model = Sequential()  # initilaizing the Sequential nature for CNN model
print(len(tk.index_word))

model.add(Embedding(len(tk.index_word), 32, input_length=100))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile( loss='binary_crossentropy',optimizer='adam',
                 metrics=['accuracy'])

print(X_train_seq_trunc)
print(np.array(y_train))
X_train_array = np.asarray(X_train_seq_trunc, dtype=np.int)
y_train_array = np.asarray(y_train, dtype=np.int)

X_test_array = np.asarray(X_test_seq_trunc, dtype=np.int)
y_test_array = np.asarray(y_test, dtype=np.int)
model.fit(X_train_array, y_train_array,validation_data=(X_test_array,y_test_array), epochs=5, batch_size=64)
model.save("tokCNN.h5")