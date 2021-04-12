import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def stem(word):
	return stemmer.stem(word.lower())

def tokenize(sentence):
	return nltk.word_tokenize(sentence)

def bag_of_words(t_sentence, words):
	stemmed_words = [stem(word) for word in t_sentence]
	bof = np.zeros(len(words), dtype=np.float32)
	for index, word in enumerate(words):
		if word in stemmed_words:
			bof[index] = 1
	return bof