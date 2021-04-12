import pandas as pd
import numpy as np

PATH = './config/'

def read_all_intents():
	df = pd.read_csv(PATH+'intents.csv')
	intents = df.to_dict('dict')
	return intents

def read_all_responses():
	df = pd.read_csv(PATH+'responses.csv')
	responses = df.to_dict('dict')
	return responses

def initialize():
	df = pd.DataFrame(list())
	df.to_csv(PATH+'intents.csv', index=False)
	df.to_csv(PATH+'responses.csv', index=False)

def create_new_intent():
	df_intents = read_all_intents()
	df_responses = read_all_responses()
	intent = input("Enter input Title of Intent:\t")
	patterns = []
	flag = True
	while(flag):
		pattern = input("Enter pattern examples (enter -1 to stop):\t")
		if(pattern == '-1'):
			flag = False
		else:
			patterns.append(pattern)
	responses = []
	flag = True
	while(flag):
		response = input("Enter response examples (enter -1 to stop):\t")
		if(response == '-1'):
			flag = False
		else:
			responses.append(response)
	print(intent, patterns, responses)

def main():
	choice = int(input("Enter:\n\t0. If you wish to initialize all the intents.\n\t1. If you wish to add another intent.\n\t2. If you wish to edit one of the intents.\nChoice:\t"))
	if choice == 0:
		initialize()
		create_new_intent()
	elif choice == 1:
		create_new_intent()
	elif choice == 2:
		df_intents = read_all_intents()
		df_responses = read_all_responses()
		print("KEYS: "+str(list(df_intents.keys())))
	else:
		print("Closing MENU")

if __name__ == '__main__':
	create_new_intent()