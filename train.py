from pprint import pprint as pprint
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPooling1D
from keras.models import Model, Input
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
import pandas
import pdb
import numpy as np

trainData = pandas.read_csv('data/clean.csv', encoding = "ISO-8859-1").fillna("clean")

testData = pandas.read_csv('data/test.csv', encoding = "ISO-8859-1").fillna("clean")


X_train = trainData["comment_text"].values
y_train = trainData[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = testData["comment_text"].values

max_features = 20000  # number of words we want to keep
maxlen = 100  # max length of the comments in the model
batch_size = 64  # batch size for the model
embedding_dims = 20  # dimension of the hidden variable, i.e. the embedding dimension

tok = Tokenizer(num_words=max_features)
tok.fit_on_texts(list(X_train) + list(X_test))
x_train = tok.texts_to_sequences(X_train)
x_test = tok.texts_to_sequences(X_test)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


comment_input = Input((maxlen,))

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
comment_emb = Embedding(max_features, embedding_dims, input_length=maxlen, 
                                embeddings_initializer="uniform")(comment_input)

# we add a GlobalMaxPooling1D, which will extract features from the embeddings
# of all words in the comment
h = GlobalMaxPooling1D()(comment_emb)

# We project onto a six-unit output layer, and squash it with a sigmoid:
output = Dense(6, activation='sigmoid')(h)

model = Model(inputs=comment_input, outputs=output)

model.compile(loss='binary_crossentropy',
                      optimizer=Adam(0.01),
                        metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=3, validation_split=0.1)
pdb.set_trace()
