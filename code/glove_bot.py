import random
import torch
import pandas as pd
import numpy as np
from glove_model import get_model
from intent_initializer import read_all_intents, read_all_responses
from GloVe_helper import GloVeLoader

PATH = './config/'
BOT_NAME = 'Bavardez'

def load_bot():
	model_details = torch.load(PATH+'model_details_GloVe.pt')
	model = get_model(model_details['input_size'], model_details['hidden_size'], model_details['output_size'])
	model.load_state_dict(model_details['model_state'])
	model.eval()

	tags = model_details['tags']

	return model, tags

def main():
	model, tags = load_bot()
	df_responses = read_all_responses()
	activation = torch.nn.Softmax(1)
	gl = GloVeLoader()

	print("Let's chat! (GloVe version) Type \"quit\" to exit.")
	while True:
		sentence = input("You:\t")
		if sentence == "quit":
			break
		embed = gl.pull_glove_embed([sentence])

		output = model(embed)
		probs = activation(output).flatten()
		predicted_label = torch.argmax(probs)
		tag = tags[predicted_label.item()]
		if probs[predicted_label]>0.5:
			if tag in list(df_responses.keys()):
				answer = random.choice(df_responses[tag])
			else:
				answer = "Sorry there's an error in OUR SYSTEM! Please re-phrase"
		else:
			answer = "I do not understand you."
		print(BOT_NAME+":\t"+answer)
	print("Thankyou for using "+BOT_NAME)		

if __name__ == '__main__':
	main()