import itertools
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import models,layers
import pickle
import json
import emoji
from keras.models import model_from_json
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
#df = pd.read_csv('data.csv')
def load_dict_contractions():
    return {
        "ain't":"is not",
        "amn't":"am not",
        "aren't":"are not",
        "can't":"cannot",
        "'cause":"because",
        "couldn't":"could not",
        "couldn't've":"could not have",
        "could've":"could have",
        "daren't":"dare not",
        "daresn't":"dare not",
        "dasn't":"dare not",
        "didn't":"did not",
        "doesn't":"does not",
        "don't":"do not",
        "e'er":"ever",
        "em":"them",
        "everyone's":"everyone is",
        "finna":"fixing to",
        "gimme":"give me",
        "gonna":"going to",
        "gon't":"go not",
        "gotta":"got to",
        "hadn't":"had not",
        "hasn't":"has not",
        "haven't":"have not",
        "he'd":"he would",
        "he'll":"he will",
        "he's":"he is",
        "he've":"he have",
        "how'd":"how would",
        "how'll":"how will",
        "how're":"how are",
        "how's":"how is",
        "I'd":"I would",
        "I'll":"I will",
        "I'm":"I am",
        "I'm'a":"I am about to",
        "I'm'o":"I am going to",
        "isn't":"is not",
        "it'd":"it would",
        "it'll":"it will",
        "it's":"it is",
        "I've":"I have",
        "kinda":"kind of",
        "let's":"let us",
        "mayn't":"may not",
        "may've":"may have",
        "mightn't":"might not",
        "might've":"might have",
        "mustn't":"must not",
        "mustn't've":"must not have",
        "must've":"must have",
        "needn't":"need not",
        "ne'er":"never",
        "o'":"of",
        "o'er":"over",
        "ol'":"old",
        "oughtn't":"ought not",
        "shalln't":"shall not",
        "shan't":"shall not",
        "she'd":"she would",
        "she'll":"she will",
        "she's":"she is",
        "shouldn't":"should not",
        "shouldn't've":"should not have",
        "should've":"should have",
        "somebody's":"somebody is",
        "someone's":"someone is",
        "something's":"something is",
        "that'd":"that would",
        "that'll":"that will",
        "that're":"that are",
        "that's":"that is",
        "there'd":"there would",
        "there'll":"there will",
        "there're":"there are",
        "there's":"there is",
        "these're":"these are",
        "they'd":"they would",
        "they'll":"they will",
        "they're":"they are",
        "they've":"they have",
        "this's":"this is",
        "those're":"those are",
        "'tis":"it is",
        "'twas":"it was",
        "wanna":"want to",
        "wasn't":"was not",
        "we'd":"we would",
        "we'd've":"we would have",
        "we'll":"we will",
        "we're":"we are",
        "weren't":"were not",
        "we've":"we have",
        "what'd":"what did",
        "what'll":"what will",
        "what're":"what are",
        "what's":"what is",
        "what've":"what have",
        "when's":"when is",
        "where'd":"where did",
        "where're":"where are",
        "where's":"where is",
        "where've":"where have",
        "which's":"which is",
        "who'd":"who would",
        "who'd've":"who would have",
        "who'll":"who will",
        "who're":"who are",
        "who's":"who is",
        "who've":"who have",
        "why'd":"why did",
        "why're":"why are",
        "why's":"why is",
        "won't":"will not",
        "wouldn't":"would not",
        "would've":"would have",
        "y'all":"you all",
        "you'd":"you would",
        "you'll":"you will",
        "you're":"you are",
        "you've":"you have",
        "Whatcha":"What are you",
        "luv":"love",
        "sux":"sucks"
        }
def remove_contractions(tweet):
  contractions = load_dict_contractions()
  return contractions[tweet.lower()] if tweet.lower() in contractions.keys() else tweet
def clean_text(tweet):
    # Remove hashtags (keeping hashtag text)
    tweet = re.sub(r'#','', tweet)
    # HTML
    tweet = re.sub(r'\&\w*;', '', tweet)
    # Remove tickers
    tweet = re.sub(r'\$\w*', '', tweet)
    # Remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    # Remove whitespaces
    tweet = re.sub(r'\s\s+','', tweet)
    tweet = re.sub(r'[ ]{2, }',' ',tweet)
    # Remove URLs, RTs and mentions(@)
    tweet=  re.sub(r'http(\S)+', '',tweet)
    tweet=  re.sub(r'http ...', '',tweet)
    tweet=  re.sub(r'(RT|rt)[ ]*@[ ]*[\S]+','',tweet)
    tweet=  re.sub(r'RT[ ]?@','',tweet)
    tweet = re.sub(r'@[\S]+','',tweet)
    # Remove words with 4 or fewer characters
    tweet = re.sub(r'\b\w{1,4}\b', '', tweet)
    # Special characteres: &, <, >
    tweet = re.sub(r'&amp;?', 'and',tweet)
    tweet = re.sub(r'&lt;','<',tweet)
    tweet = re.sub(r'&gt;','>',tweet)
    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    tweet= ''.join(c for c in tweet if c <= '\uFFFF')
    tweet = tweet.strip()
    # Remove misspelling words
    tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))
    # Remove emoji
    tweet = emoji.demojize(tweet)
    tweet = tweet.replace(":"," ")
    tweet = ' '.join(tweet.split())
    tweet = re.sub("([^\x00-\x7F])+"," ",tweet)
    # Remove mojibake (also extra spaces)
    tweet = ' '.join(re.sub("[^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a]", " ", tweet).split())
    return tweet
df["text"] = df["text"].apply(remove_contractions)
df["text"] = df["text"].apply(clean_text)

#corpus=[]
#for index in range(len(df["headlines"])):
#   corpus.append(preprocessing.text_preprocessing(df["headlines"][index]))


df.drop_duplicates(subset=["text"], inplace=True)
df.dropna(inplace=True)
print(df.head)
voc_size=5000
sent_length=100
X_train, X_test, y_train, y_test = train_test_split(df.text, df.target, test_size=0.3, random_state=37)
tk = Tokenizer()
tk.fit_on_texts(X_train)
X_train_seq = tk.texts_to_sequences(X_train)
X_test_seq = tk.texts_to_sequences(X_test)

X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=100)
X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=100)
model = Sequential()  # initilaizing the Sequential nature for CNN model
print(len(tk.index_word))

model.add(Embedding(len(tk.index_word)+3, 32, input_length=100))
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


with open('tokenizer.pickle', 'wb') as handle:
      pickle.dump(tk, handle, protocol=pickle.HIGHEST_PROTOCOL)
model_LSTM_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_LSTM_json)
model.save_weights("model.h5")
print("Saved model to disk")

#pickle.dump(model,open('model3.pkl','wb'))
def Save_Tokenizer():
    return tk
    