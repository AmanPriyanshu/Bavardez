import pandas as pd
import numpy as np
from intent_initializer import read_all_intents, read_all_responses
from preprocessing import stem, tokenize, bag_of_words
import torch
from model import get_model

def read_everything():
	df_responses = read_all_responses()
	df_patterns = read_all_intents()
	
	assert len(list(df_responses.keys())) == len(list(df_patterns.keys())), "Patterns and Responses should have the same Tags"

	intents = list(df_responses.keys())
	patterns = df_responses.values()
	responses = df_responses.values()
	all_patterns = sum(patterns, [])
	all_words = []
	xy = sum([[(tokenize(sentence), tag) for sentence in sentences] for tag, sentences in df_responses.items()], [])

	for sentence in all_patterns:
		all_words.extend(tokenize(sentence))
	all_words = sorted(set([stem(w) for w in all_words]))
	tags = sorted(intents)

	train_x, train_y = [], []
	for (sentence, tag) in xy:
		bof = bag_of_words(sentence, all_words)
		train_x.append(bof)
		train_y.append(tags.index(tag))
	train_x, train_y = np.array(train_x), np.array(train_y)
	return train_x, train_y

class ChatDataset(torch.utils.Dataset):
	def __init__(self):
		self.train_x, self.train_y = read_everything()
		self.n_samples = len(self.train_x)

	def __getitem__(self, index):
		return self.train_x[index], self.y_train[index]

	def __len__(self):
		return self.n_samples

if __name__ == '__main__':
	dataset = ChatDataset()
	train_loader = torch.utils.DataLoader(dataset=dataset, batch_size=8, shuffle=True)