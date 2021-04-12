import random
import torch
import pandas as pd
import numpy as np
from model import get_model
from intent_initializer import read_all_intents, read_all_responses
from preprocessing import stem, tokenize, bag_of_words

PATH = './config/'

def load_bot():
	model_details = torch.load(PATH+'model_details.pt')
	model = get_model(model_details['input_size'], model_details['hidden_size'], model_details['output_size'])
	model.load_state_dict(model_details['model_state'])
	model.eval()

	tags = model_details['tags']
	all_words = model_details['all_words']

if __name__ == '__main__':
	load_bot()