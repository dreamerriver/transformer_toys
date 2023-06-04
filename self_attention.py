import torch
from torch import nn
from torch.nn import LayerNorm
from torch.nn import Dropout
class SelfAttention(nn.Module):
    def __init__(self,input_dim,mask=None,drop_rate=0.5):
        super().__init__()
        self.Q = nn.Linear(input_dim,input_dim)
        self.K = nn.Linear(input_dim,input_dim)
        self.V = nn.Linear(input_dim,input_dim)
        self.layernorm = LayerNorm(input_dim)
        self.temperature = 1.0
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = Dropout(drop_rate)
        self.mask = mask

    def forward(self,x):
        q = self.layernorm(self.Q(x))
        k = self.layernorm(self.K(x))
        v = self.layernorm(self.V(x))
        k = torch.transpose(k,-1,-2)
        att_val = torch.matmul(q,k)
        if self.mask:
            att_val = att_val * self.mask
        att_val = self.softmax(att_val)
        att_val = self.dropout(att_val)
        x = torch.matmul(att_val,v)
        return x

if __name__ == '__main__':
    atten = SelfAttention(768,mask=True)
    x = torch.randn([8,128,768])
    x = atten(x)
    #print(x.shape)

