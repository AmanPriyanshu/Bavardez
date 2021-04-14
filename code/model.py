import torch

def get_model(input_size, hidden_size, num_classes):
	model = torch.nn.Sequential(
		torch.nn.Linear(input_size, hidden_size),
		torch.nn.ReLU(),
		torch.nn.Linear(hidden_size, hidden_size),
		torch.nn.ReLU(),
		torch.nn.Linear(hidden_size, num_classes)
		)
	return model