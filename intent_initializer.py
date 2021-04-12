import pandas as pd
import numpy as np

def read_all_intents():
	df = pd.read_csv('intents.csv')
	intents = df.to_dict('dict')
	return intents

def read_all_responses():
	df = pd.read_csv('responses.csv')
	responses = df.to_dict('dict')
	return responses

def create_new_intent():
	choice = int(input("Enter \n\t0. If you wish to initialize all the intents.\n\t1. If you wish to add another intent.\n\t2. If you wish to edit one of the intents."))
	if choice == 0:
		df = pd.DataFrame(list())
		df.to_csv('intents.csv', index=False)
		df.to_csv('responses.csv', index=False)
	elif choice == 1:
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
	elif choice == 2:
		df_intents = read_all_intents()
		df_responses = read_all_responses()
		print("KEYS: "+str(list(df_intents.keys())))
	else:
		print("Closing MENU")

if __name__ == '__main__':
	create_new_intent()