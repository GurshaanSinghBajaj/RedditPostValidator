# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,LSTM,Bidirectional, Embedding, SpatialDropout1D
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import re

MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text

def text_cleaner(text):
    text = text.map(lambda x: re.sub('[,\.!?]', '', x))
    text = text.map(lambda x: x.lower())
    text = text.map(lambda x: clean_text(x))
    return text

# Importing Data
body = pd.read_csv("./fnc-1/train_bodies.csv")
stances = pd.read_csv("./fnc-1/train_stances.csv")
data = pd.merge(stances,body,on="Body ID")
X = data[["Headline","articleBody"]]
y = data["Stance"]
X["Headline"] = text_cleaner(X["Headline"])
X["articleBody"] = text_cleaner(X["articleBody"])

#Cleaning Data
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(pd.concat([X["articleBody"],X["Headline"]]).values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
X = tokenizer.texts_to_sequences(pd.concat([X["articleBody"],X["Headline"]]).values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)
X_new=[]
for i in range(0,int(len(X)/2)):
    X_new.append([X[i]+X[i + int(len(X)/2)]])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(np.array(y))

# Train, test and validation split
X_train,X_val,y_train,y_val = train_test_split(X_new,y,test_size=0.2)
X_train=np.array(X_train).reshape((len(X_train),250))
X_val=np.array(X_val).reshape((len(X_val),250))

# Creating Model
# optimizer = SGD(learning_rate=0.001,momentum=0.0, nesterov = False)
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(128, activation = "relu",dropout=0.2, recurrent_dropout=0.2))
# model.add(LSTM(64, activation = "tanh",dropout=0.2, recurrent_dropout=0.2))
# model.add(LSTM(128, activation = "relu",dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="tanh"))
model.add(Dense(32,activation="relu"))
model.add(Dense(64,activation="tanh"))
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dense(4,activation="softmax"))
model.summary()
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# Training Model
model.fit(X_train, y_train,batch_size=128,
          epochs=10,validation_data=(X_val,y_val),shuffle=True)

y_pred = model.predict(X_val)
for i in range(len(y_pred)):
    temp = [0,0,0,0]
    temp[np.argmax(y_pred[i])] =  1
    y_pred[i]=temp
print(multilabel_confusion_matrix(y_val, y_pred))
