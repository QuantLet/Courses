from torch import nn
import torch

class CNN(nn.Module):
    def __init__(self, seq_len, batch_size, training):
        super(CNN, self).__init__()
        self.layer = 3
        self.seq_len = seq_len
        self.conv = nn.Conv2d(in_channels=3, out_channels = 2, kernel_size=(1,3), stride = 1, padding = 'valid')
        self.dropout = nn.Dropout2d(0.2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=20, kernel_size=(1,seq_len-2), stride = 1, padding = 'valid')
        if training == 'buffer':
            self.conv3 = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=1, padding = 'valid')
        else:
            self.conv3 = nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1, padding = 'valid')
        self.last_activ = nn.Softmax(dim = 1)
        self.dense = nn.Linear(8,9)
    
    def forward(self, x, lastw = None):
        self.means = x.mean(-1, keepdim=True).detach()
        x = x - self.means
        self.stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
        x /= self.stdev
        
        x = self.conv(x)
        x = self.relu(x) 
        x =self.dropout(x)

        x = self.conv2(x)
        x = self.relu(x)
        x =self.dropout(x)
        if lastw != None :
            x = torch.concatenate((lastw[:,None,:,None],x),dim=1)
        # print(x)
        x = self.conv3(x)
        x = self.relu(x)
        # print(x)
        x = x.flatten(start_dim=1)
        x = self.last_activ(x)
        x = torch.cat((torch.zeros_like(x)[:,0:1], x), dim = 1)
        return x