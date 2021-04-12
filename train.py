import pandas as pd
import numpy as np
from intent_initializer import read_all_intents, read_all_responses
from preprocessing import stem, tokenize, bag_of_words

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
	tags = sorted(tags)

	train_x, train_y = [], []
	for (sentence, tag) in xy:
		bof = bag_of_words()

if __name__ == '__main__':
	read_everything()