import torch

class LSTM_ANN(torch.nn.Module):
	def __init__(self, input_size, rnn_hidden_size, stacks, output_size, n_words, hidden_size):
		super(LSTM_ANN, self).__init__()
		self.rnn = torch.nn.RNN(input_size, rnn_hidden_size, 3, batch_first=True)
		self.linear_first = torch.nn.Linear(rnn_hidden_size*n_words, hidden_size)
		self.linear = torch.nn.Linear(hidden_size, hidden_size)
		self.output_layer = torch.nn.Linear(hidden_size, output_size)
		self.flatten = torch.nn.Flatten()
		self.relu = torch.nn.ReLU()

	def forward(self, embed):
		out, _ = self.rnn(embed)
		out = self.flatten(out)
		out = self.linear_first(out)
		out = self.relu(out)
		out = self.linear(out)
		out = self.relu(out)
		out = self.output_layer(out)
		return out
		


def get_model(input_size, hidden_size, output_size, rnn_hidden_size=10, stacks=3, n_words=10):
	model = LSTM_ANN(input_size, rnn_hidden_size, stacks, output_size, n_words, hidden_size)
	return model
