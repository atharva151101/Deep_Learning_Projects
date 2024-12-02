import torch
from torch import nn
import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(42)


class ClassificationImage(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear1 = nn.Linear(cfg.input_num, cfg.num_hidden1)
        self.linear2 = nn.Linear(cfg.num_hidden1, cfg.num_hidden2)
        self.linear3 = nn.Linear(cfg.num_hidden2, cfg.output_num)
        self.activation = nn.Tanh()
        self.soft = nn.Softmax()

    def forward(self, x):
        h1 = self.activation(self.linear1(x))
        h2 = self.activation(self.linear2(h1))
        y = self.linear3(h2)
        return self.soft(y)

