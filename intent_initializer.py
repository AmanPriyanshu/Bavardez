import pandas as pd
import numpy as np

def read_all_intents():
	df = pd.read_csv('intents.csv')
	intents = df.to_dict('dict')

def create_new_intent():
	pass