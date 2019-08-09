import torch
import torch.nn as nn

class Config(object):
    def __init__(self):
        self.path = './data/ml-tag'
        self.batch_size = 4096
        self.GPU_available = False 
        self.embedding_dim = 70
        self.epoch = 100
        self.lr = 1e-2