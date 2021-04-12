import pandas as pd
import numpy as np

PATH = './config/'

def read_all_intents():
	df = pd.read_csv(PATH+'intents.csv')
	intents = df.to_dict('dict')
	intents = {key: [i for i in list(value.values()) if str(i)!='nan'] for key, value in intents.items()}
	return intents

def read_all_responses():
	df = pd.read_csv(PATH+'responses.csv')
	responses = df.to_dict('dict')
	responses = {key: [i for i in list(value.values()) if str(i)!='nan'] for key, value in responses.items()}
	return responses

def initialize():
	df = pd.DataFrame(list())
	df.to_csv(PATH+'intents.csv', index=False)
	df.to_csv(PATH+'responses.csv', index=False)

def create_new_intent():
	intent = input("Enter input Title of Intent:\t")
	patterns = []
	flag = True
	while(flag):
		pattern = input("Enter pattern examples (enter -1 to stop):\t")
		if(pattern == "-1"):
			flag = False
		else:
			patterns.append(pattern)
	print("\nNow recording responses...\n")
	responses = []
	flag = True
	while(flag):
		response = input("Enter response examples (enter -1 to stop):\t")
		if(response == '-1'):
			flag = False
		else:
			responses.append(response)
	print("\nCompleted and Saved both Intents and Responses.")
	
	return intent, patterns, responses

def main():
	while(1):
		choice = int(input("Enter:\n\t0. If you wish to initialize all the intents.\n\t1. If you wish to add another intent.\n\t2. If you wish to edit one of the intents.\nChoice:\t"))
		if choice == 0:
			initialize()
			intent, patterns, responses = create_new_intent()
			patterns = pd.DataFrame({intent: patterns})
			responses = pd.DataFrame({intent: responses})
			patterns.to_csv(PATH+'intents.csv', index=False)
			responses.to_csv(PATH+'responses.csv', index=False)
		elif choice == 1:
			df_intents = read_all_intents()
			df_responses = read_all_responses()
			intent, patterns, responses = create_new_intent()
			df_intents.update({intent: patterns})
			df_responses.update({intent: responses})
			df_responses = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in df_responses.items() ]))
			df_intents = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in df_intents.items() ]))
			df_intents.to_csv(PATH+'intents.csv', index=False)
			df_responses.to_csv(PATH+'responses.csv', index=False)

		elif choice == 2:
			df_intents = read_all_intents()
			df_responses = read_all_responses()
			print("KEYS: "+str(list(df_intents.keys())))
		else:
			print("Closing MENU")
			break
		print("\n\n")

if __name__ == '__main__':
	main()