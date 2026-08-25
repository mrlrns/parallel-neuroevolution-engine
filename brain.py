import torch
import torch.nn as nn 




class Brain(nn.Module):
    def __init__(self, input_size, output_size):
        super(Brain,self).__init__()
        self.layer1=nn.Linear(input_size,64)
        self.layer2=nn.Linear(64,64)
        self.layer3=nn.Linear(64,output_size)

    def forward(self,x):
        x=torch.relu(self.layer1(x))
        x=torch.relu(self.layer2(x))
        x=torch.tanh(self.layer3(x))
        return x
    
    
    
    
