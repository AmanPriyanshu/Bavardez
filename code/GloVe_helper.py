import numpy as np
import torch
import nltk
from nltk.stem import WordNetLemmatizer
import string

class GloVeLoader:
	def __init__(self, word_limit=10, glove_path='./GloVe/glove.6B.50d.txt'):
		self.glove_path = glove_path
		self.word_limit = word_limit
		self.embeddings_dict = {}
		self.glove_loader()
		self.wordnet_lemmatizer = WordNetLemmatizer()
		self.punctuation = string.punctuation

	def glove_loader(self):
		with open(self.glove_path, 'r', encoding="utf-8") as f:
			for line in f:
				values = line.split()
				word = values[0]
				vector = np.asarray(values[1:], "float32")
				self.embeddings_dict[word.lower()] = vector

	def pull_glove_embed(self, sentences):
		sample = []
		for sentence in sentences:
			vec = []
			for w in nltk.word_tokenize(sentence)[:self.word_limit]:
				if w not in self.punctuation:
					try:
						vec.append(self.embeddings_dict[w.lower()])
					except:
						try:
							vec.append(self.embeddings_dict[self.wordnet_lemmatizer.lemmatize(w.lower())])
						except:
							vec.append(np.zeros(50))
			vec += [np.zeros(50) for _ in range(self.word_limit - len(vec))]
			sample.append(vec)
		sample = np.stack(sample)
		sample = sample.astype(np.float32)
		sample = torch.from_numpy(sample)
		return sample

if __name__ == '__main__':
	gl = GloVeLoader()
	x = gl.pull_glove_embed(['hello how are you? ornfeorjgne', 'thank you! for telling me about the items I can buy here. good bye'])
	print(x)