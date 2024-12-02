import torch
from torch import nn
torch.manual_seed(42)


class AutoEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.encoder=nn.Sequential(nn.Linear(cfg.input_num,cfg.num_ae_hidden1),
        			   nn.ReLU(),
        			   nn.Linear(cfg.num_ae_hidden1,cfg.num_ae_hidden2),
        			   nn.ReLU(),
        			   nn.Linear(cfg.num_ae_hidden2, cfg.num_ae_output))
        self.decoder=nn.Sequential(nn.Linear(cfg.num_ae_output,cfg.num_ae_hidden2),
        			   nn.ReLU(),
        			   nn.Linear(cfg.num_ae_hidden2, cfg.num_ae_hidden1),
        			   nn.ReLU(),
        			   nn.Linear(cfg.num_ae_hidden1, cfg.input_num))
       
    
    def forward(self, x):
        code=self.encoder(x)
        return self.decoder(code)

