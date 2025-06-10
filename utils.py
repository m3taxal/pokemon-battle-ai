import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' # force cpu due to overhead of moving data from cpu to gpu