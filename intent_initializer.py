import pandas as pd
import numpy as np

def read_all_intents():
	df = pd.read_csv('intents.csv')
	intents = df.to_dict('dict')

def create_new_intent():
	choice = int(input("Enter \n\t0. If you wish to initialize all the intents.\n\t1. If you wish to add another intent.\n\t2. If you wish to edit one of the intents."))
	if choice == 0:
		df = pd.DataFrame(list())
		df.to_csv('intents.csv', index=False)
		df.to_csv('responses.csv', index=False)