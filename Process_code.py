import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,Conv1D,Flatten,Concatenate
from keras.models import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint

TRAIN_DATA_FILE= 'train.csv'
TEST_DATA_FILE= 'test.csv'

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

embed_size = 100 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a comment to use


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = LSTM(4, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)(x)
x = Conv1D(16,4,activation='relu')(x)#LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(x)
x = Flatten()(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer=optimizers.rmsprop(lr = 0.001,decay = 1e-06), metrics=['accuracy'])
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_t, y, batch_size=32, epochs=10,callbacks=callbacks_list, verbose=1, validation_split=0.1)

y_te = model.predict(X_te)
Submit = pd.DataFrame(test.id,columns=['id'])
Submit2 = pd.DataFrame(y_te,columns=list_classes)
Submit = pd.concat([Submit,Submit2],axis=1)
