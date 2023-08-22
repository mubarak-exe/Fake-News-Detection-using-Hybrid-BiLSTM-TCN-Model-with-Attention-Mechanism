# -*- coding: utf-8 -*-
"""CNN_LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15qBW3B3za5Ey_QVE2GpELZr1FIM3av9C

# Importing the required Libraries
"""

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Conv1D, MaxPool1D, Input, LSTM, BatchNormalization, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.dtypes import uint8, float32
import pickle

from tensorflow.random import set_seed
set_seed(5)

from google.colab import drive
drive.mount('/content/drive')

"""# Data Loading and Preprocessing"""

true = pd.read_csv('/content/drive/MyDrive/data_set_1/ISOT Fake News Dataset/True.csv')
fake = pd.read_csv('/content/drive/MyDrive/data_set_1/ISOT Fake News Dataset/Fake.csv')

# add 1 for label for true and 0 fro fake
true["label"] = 1
fake['label'] = 0

# Combine both dataframes and shuffle
input_data = pd.concat( [true,fake] )
input_data = input_data.sample(frac = 1)

# remove website url and ip
input_data['text']= input_data['text'].apply(lambda x: re.sub(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})", "", x))
input_data['text']= input_data['text'].apply(lambda x: re.sub(r"^(?!mailto:)(?:(?:http|https|ftp)://)(?:\\S+(?::\\S*)?@)?(?:(?:(?:[1-9]\\d?|1\\d\\d|2[01]\\d|22[0-3])(?:\\.(?:1?\\d{1,2}|2[0-4]\\d|25[0-5])){2}(?:\\.(?:[0-9]\\d?|1\\d\\d|2[0-4]\\d|25[0-4]))|(?:(?:[a-z\\u00a1-\\uffff0-9]+-?)*[a-z\\u00a1-\\uffff0-9]+)(?:\\.(?:[a-z\\u00a1-\\uffff0-9]+-?)*[a-z\\u00a1-\\uffff0-9]+)*(?:\\.(?:[a-z\\u00a1-\\uffff]{2,})))|localhost)(?::\\d{2,5})?(?:(/|\\?|#)[^\\s]*)?$", "", x))
input_data['text']= input_data['text'].apply(lambda x: re.sub(r"^((25[0-5]|(2[0-4]|1[0-9]|[1-9]|)[0-9])(\.(?!$)|$)){4}$", "", x))

# Remove Stopwords
import nltk
nltk.download('stopwords')
stopwords=stopwords.words('english')
input_data['text'] = input_data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

#STEMMING
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
nltk.download('punkt')

porter = PorterStemmer()
# for word in input_data['text']:
#     print(porter.stem(word))
input_data['text'] = input_data['text'].apply(lambda x: ' '.join([porter.stem(y) for y in x.split()]))

"""Mapping Text to Vectors"""

pip install keras-preprocessing

# Tockenization
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding

tokenizer = Tokenizer(num_words=9999999999)
tokenizer.fit_on_texts(input_data['text'])
sequences = tokenizer.texts_to_sequences(input_data['text'])
word_index = tokenizer.word_index

len(sequences)

import tensorflow as tf
sequences=tf.keras.preprocessing.sequence.pad_sequences(
    sequences,
    maxlen=100,
    dtype='int32',
    padding='post',
    truncating='pre',
    value=0.0
)

GLOVE_DIR = "data"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, '/content/drive/MyDrive/data_set_1/ISOT Fake News Dataset/glove.6B.300d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    #print(values[1:])
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors in Glove.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            300,
                            weights=[embedding_matrix],
                            input_length=100)

path1 = '/content/drive/MyDrive/Colab Notebooks/Project_Models/Processed_Data/New'
pickle.dump(embedding_layer, open(path1+'i100embedding_layer.pkl', 'wb'))

pickle.dump(embedding_layer, open(path1+'i100embedding_layer.pkl', 'wb'))

"""# SPLITTING THE DATA

"""

data=sequences
label= input_data["label"]
x_train, x_test, y_train, y_test = train_test_split( data, label, test_size=0.20, random_state=42)
x_test, x_val, y_test, y_val = train_test_split( x_test, y_test, test_size=0.50, random_state=42)
print('Size of train, validation, test:', len(y_train), len(y_val), len(y_test))

print('real & fake news in train,valt,test:')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))
print(y_test.sum(axis=0))

# pickle.dump(x_train, open(path+'x_train.pkl', 'wb'))
# pickle.dump(y_train, open(path+'y_train.pkl', 'wb'))
# pickle.dump(y_test, open(path+'y_test.pkl', 'wb'))
# pickle.dump(x_test, open(path+'x_test.pkl', 'wb'))
# pickle.dump(x_val, open(path+'x_val.pkl', 'wb'))
# pickle.dump(y_val, open(path+'y_val.pkl', 'wb'))

"""# MODEL"""

path = '/content/drive/MyDrive/Colab Notebooks/Project_Models/Processed_Data/'
x_train=pickle.load(open(path+'x_train.pkl', 'rb'))
y_train=pickle.load(open(path+'y_train.pkl', 'rb'))
y_test=pickle.load(open(path+'y_test.pkl', 'rb'))
x_test=pickle.load(open(path+'x_test.pkl', 'rb'))
x_val=pickle.load(open(path+'x_val.pkl', 'rb'))
y_val=pickle.load(open(path+'y_val.pkl', 'rb'))
embedding_layer = pickle.load(open(path1+'i100embedding_layer.pkl', 'rb'))

i = Input(100, dtype=uint8)
x = embedding_layer(i)
x = Conv1D(128, 5, activation='relu')(x)
x= Dropout(0.30)(x)
x= BatchNormalization()(x)
x = MaxPool1D()(x)
x = LSTM(32, activation='linear')(x)
x= Dropout(0.30)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=[i], outputs=output)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

class myCallback(Callback):
  def on_epoch_end(self, epoch, logs={}):
    acc = logs.get('accuracy')
    val_acc = logs.get('val_accuracy')
    if (epoch % 5 == 0) and (epoch != 0):
      model_name = f'CNN-LSTM_e{epoch}'  # add model name (name_) as required
      model_path = '/content/drive/MyDrive/Colab Notebooks/Project_Models/CNN_LSTM_Models'  # add model path as required
      self.model.save(os.path.join(model_path, model_name))

callback = myCallback()

BATCH_SIZE = 64
EPOCHS = 2
TRAINING_STEPS = len(x_train) //  BATCH_SIZE
VALIDATION_STEPS = len(x_val) // BATCH_SIZE

history = model.fit(x_train,y_train,
                    steps_per_epoch= TRAINING_STEPS,
                    validation_data=[x_val,y_val],
                    validation_steps=VALIDATION_STEPS,
                    epochs=EPOCHS,
                    callbacks=[callback],
                    verbose='auto')

pickle.dump(history, open(path+'history_CNN_LSTM.pkl', 'wb'))

history.history["accuracy"]

"""# Metrics and Graphs

"""

# # Training History
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'], )
# plt.title('Model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

model=load_model('/content/drive/MyDrive/Colab Notebooks/Project_Models/CNN_LSTM_Models/CNN-LSTM_e35')

y_pred=model.predict(x_test)
y_pred = np.squeeze(y_pred)
y_pred

p = lambda t : 1 if t>=0.5 else 0
y_pred=np.vectorize(p)(y_pred)

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

dot_img_file = '/tmp/model_1.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)