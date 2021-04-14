import torch

def get_model(input_size, hidden_size, output_size, rnn_hidden_size=25):
	model = torch.nn.Sequential(
		torch.nn.RNN(input_size, rnn_hidden_size, 3),
		)
	return model
