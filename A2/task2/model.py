import torch
from torch import nn
torch.manual_seed(42)


class AE1(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.encoder1=nn.Sequential(nn.Linear(cfg.ae1_input,cfg.ae1_hidden1),
        			   nn.ReLU(),
        			   nn.Linear(cfg.ae1_hidden1,cfg.ae1_hidden2),
        			   nn.ReLU(),
        			   nn.Linear(cfg.ae1_hidden2, cfg.ae1_output))
        self.decoder1=nn.Sequential(nn.Linear(cfg.ae1_output,cfg.ae1_hidden2),
        			   nn.ReLU(),
        			   nn.Linear(cfg.ae1_hidden2, cfg.ae1_hidden1),
        			   nn.ReLU(),
        			   nn.Linear(cfg.ae1_hidden1, cfg.ae1_input))
       
    
    def forward(self, x):
        code=self.encoder1(x)
        return self.decoder1(code)

class AE2(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder2=nn.Sequential(nn.Linear(cfg.ae2_input,cfg.ae2_hidden1),
        			   nn.ReLU(),
        			   nn.Linear(cfg.ae2_hidden1,cfg.ae2_hidden2),
        			   nn.ReLU(),
        			   nn.Linear(cfg.ae2_hidden2, cfg.ae2_output))
        self.decoder2=nn.Sequential(nn.Linear(cfg.ae2_output,cfg.ae2_hidden2),
        			   nn.ReLU(),
        			   nn.Linear(cfg.ae2_hidden2, cfg.ae2_hidden1),
        			   nn.ReLU(),
        			   nn.Linear(cfg.ae2_hidden1, cfg.ae2_input))
       
    
    def forward(self, x):
        code=self.encoder2(x)
        return self.decoder2(code)
    
class AE3(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder3=nn.Sequential(nn.Linear(cfg.ae3_input,cfg.ae3_hidden1),
        			   nn.ReLU(),
        			   nn.Linear(cfg.ae3_hidden1,cfg.ae3_hidden2),
        			   nn.ReLU(),
        			   nn.Linear(cfg.ae3_hidden2, cfg.ae3_output))
        self.decoder3=nn.Sequential(nn.Linear(cfg.ae3_output,cfg.ae3_hidden2),
        			   nn.ReLU(),
        			   nn.Linear(cfg.ae3_hidden2, cfg.ae3_hidden1),
        			   nn.ReLU(),
        			   nn.Linear(cfg.ae3_hidden1, cfg.ae3_input))
       
    
    def forward(self, x):
        code=self.encoder3(x)
        return self.decoder3(code)
    
class stacked_autoencoder(torch.nn.Module):
    def __init__(self, cfg,AE1,AE2,AE3):
        super().__init__()
        self.encoder1=AE1.encoder1
        self.encoder2=AE2.encoder2
        self.encoder3=AE3.encoder3
        self.linear= nn.Linear(cfg.ae3_output,cfg.output_num)
        self.soft = nn.Softmax()

    def forward(self, x):
        x=self.encoder3(self.encoder2(self.encoder1(x)))
        y=self.soft(self.linear(x))
        return y

