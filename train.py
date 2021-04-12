import pandas as pd
import numpy as np
from intent_initializer import read_all_intents, read_all_responses
from preprocessing import stem, tokenize, bag_of_words
import torch
from model import get_model
from tqdm import trange

PATH = './config/'

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
	tags = sorted(intents)

	train_x, train_y = [], []
	for (sentence, tag) in xy:
		bof = bag_of_words(sentence, all_words)
		train_x.append(bof)
		train_y.append(tags.index(tag))
	train_x, train_y = np.array(train_x), np.array(train_y)
	train_y = train_y.astype(np.int64)
	return train_x, train_y, tags, all_words

class ChatDataset(torch.utils.data.Dataset):
	def __init__(self):
		self.train_x, self.train_y, self.tags, self.all_words = read_everything()
		self.n_samples = len(self.train_x)

	def __getitem__(self, index):
		return self.train_x[index], self.train_y[index]

	def __len__(self):
		return self.n_samples

def train_instance(hidden_size=8, lr=0.001, num_epochs=500):
	dataset = ChatDataset()
	train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=8, shuffle=True)
	model = get_model(len(dataset.all_words), hidden_size, len(dataset.tags))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	bar = trange(num_epochs)
	for epoch in bar:
		running_loss, running_accuracy = 0, 0
		for (words, labels) in train_loader:
			words = words.to(device)
			labels = labels.to(device)

			outputs = model(words)
			loss = criterion(outputs, labels)

			predictions = torch.argmax(outputs, 1)
			accuracy = torch.mean((predictions == labels).float())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item()
			running_accuracy += accuracy.item()
		running_loss /= len(train_loader)
		running_accuracy /= len(train_loader)
		bar.set_description(str({'epoch':epoch+1, 'loss':round(running_loss, 4), 'acc': round(running_accuracy, 4)}))
	bar.close()

	model_details = {
	"model_state": model.state_dict(),
	"input_size": len(dataset.all_words),
	"output_size": len(dataset.tags),
	"hidden_size": hidden_size,
	"all_words": dataset.all_words,
	"tags": dataset.tags
	}

	torch.save(model_details, PATH+'model_details.pt')

	print("Training has been Completed. Model has been saved at \""+PATH+'model_details.pt"')

if __name__ == '__main__':
	train_instance()