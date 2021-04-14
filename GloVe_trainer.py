import torch
from GloVe_helper import GloVeLoader
import pandas as pd
import numpy as np
from intent_initializer import read_all_intents, read_all_responses
from glove_model import get_model

PATH = './config/'

def read_everything():
	df_responses = read_all_responses()
	df_patterns = read_all_intents()
	tags = list(df_responses.keys())
	tags = sorted(tags)

	assert len(list(df_responses.keys())) == len(list(df_patterns.keys())), "Patterns and Responses should have the same Tags"

	xy = [[(tag, sentence) for sentence in sentences] for tag,sentences in df_patterns.items()]
	xy = np.array(sum(xy, []))
	gl = GloVeLoader()
	x_train = gl.pull_glove_embed(xy.T[1])
	y_train = np.array([tags.index(tag) for tag in xy.T[0]])
	return x_train, y_train, tags

class ChatDataset(torch.utils.data.Dataset):
	def __init__(self):
		self.train_x, self.train_y, self.tags = read_everything()
		self.n_samples = len(self.train_x)

	def __getitem__(self, index):
		return self.train_x[index], self.train_y[index]

	def __len__(self):
		return self.n_samples

def train_instance(hidden_size=8, lr=0.001, num_epochs=500):
	dataset = ChatDataset()
	train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=8, shuffle=True)
	model = get_model(50, hidden_size, len(dataset.tags))
	print(model)
	exit()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	bar = trange(num_epochs)
	for epoch in bar:
		running_loss, running_accuracy = 0, 0
		for (embeds, labels) in train_loader:
			embeds = embeds.to(device)
			labels = labels.to(device)

			outputs = model(embeds)
			print(outputs.shape)
			exit()
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
	"input_size": 50,
	"output_size": len(dataset.tags),
	"hidden_size": hidden_size,
	"all_words": dataset.all_words,
	"tags": dataset.tags
	}

	torch.save(model_details, PATH+'model_details_GloVe.pt')

	print("Training has been Completed. Model has been saved at \""+PATH+'model_details_GloVe.pt"')


if __name__ == '__main__':
	train_instance(num_epochs=10)