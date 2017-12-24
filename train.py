from pprint import pprint as pprint
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
import pandas


HEADER = "id,comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate"

trainData = pandas.read_csv('data/clean.csv', encoding = "ISO-8859-1").fillna("clean")

testData = pandas.read_csv('data/test.csv', encoding = "ISO-8859-1").fillna("clean")

pprint(trainData.head())

pprint(testData.head())
