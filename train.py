import pandas as pd
import numpy as np
from intent_initializer import read_all_intents, read_all_responses

def read_everything():
	df_responses = read_all_responses()
	df_patterns = read_all_patterns()
	
	assert len(list(df_responses.keys())) == len(list(df_patterns.keys())), "Patterns and Responses should have the same Tags"

	intents = list(df_responses.keys())