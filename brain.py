import torch.nn as nn 


class Brain(nn.Module):
    def __init__(self, input_size, output_size):
        super(Brain,self).__init__()
        self.layer1=nn.Linear(input_size,64)
        self.layer2=nn.Linear(64,64)
        self.layer3=nn.Linear(64,output_size)

    def forward(self,x):
        x=nn.ReLU()(self.layer1(x))
        x=nn.ReLU()(self.layer2(x))
        x=nn.Tanh()(self.layer3(x))
        return x
    
    
    
    
