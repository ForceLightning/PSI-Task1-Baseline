import torch

CUDA = True if torch.cuda.is_available() else False
DEVICE = torch.device("cuda:0" if CUDA else "cpu")
FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if CUDA else torch.LongTensor
