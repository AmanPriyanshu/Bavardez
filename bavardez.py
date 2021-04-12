import random
import torch
import pandas as pd
import numpy as np
from model import get_model
from intent_initializer import read_all_intents, read_all_responses
from preprocessing import stem, tokenize, bag_of_words

PATH = './config/'
BOT_NAME = 'Bavardez'

def load_bot():
	model_details = torch.load(PATH+'model_details.pt')
	model = get_model(model_details['input_size'], model_details['hidden_size'], model_details['output_size'])
	model.load_state_dict(model_details['model_state'])
	model.eval()

	tags = model_details['tags']
	all_words = model_details['all_words']

	return model, tags, all_words

def main():
	model, tags, all_words = load_bot()

	print("Let's chat! Type \"quit\" to exit.")
	while True:
		sentence = input("You:\t")
		if sentence == "quit":
			break
		sentence = tokenize(sentence)
		bof = bag_of_words(sentence, all_words)
		bof = np.expand_dims(bof, axis=0)
		bof = torch.from_numpy(bof)

		output = model(bof)
		predicted_label = torch.argmax(output)
		tag = tags[predicted_label.item()]
		print(tag)
		exit()

if __name__ == '__main__':
	main()