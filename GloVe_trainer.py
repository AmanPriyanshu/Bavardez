import torch
from GloVe_helper import GloVeLoader
import pandas as pd
import numpy as np
from intent_initializer import read_all_intents, read_all_responses

PATH = './config/'

def read_everything():
	df_responses = read_all_responses()
	df_patterns = read_all_intents()
	
	assert len(list(df_responses.keys())) == len(list(df_patterns.keys())), "Patterns and Responses should have the same Tags"

	xy = [[(tag, sentence) for sentence in sentences] for tag,sentences in df_patterns.items()]
	print(xy)

class ChatDataset(torch.utils.data.Dataset):
	def __init__(self):
		self.train_x, self.train_y, self.tags, self.all_words = read_everything()
		self.n_samples = len(self.train_x)

	def __getitem__(self, index):
		return self.train_x[index], self.train_y[index]

	def __len__(self):
		return self.n_samples

if __name__ == '__main__':
	read_everything()