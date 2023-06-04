import torch
from torch import nn
from torch.nn import  Dropout
from torch.nn import Softmax
from mask_generator import encoder_mask
from mask_generator import decoder_mask
from mask_generator import cross_mask

class MultiheadSelfAttention(nn.Module):
    def __init__(self,input_dim,heads,drop_rate=0.5,is_decoder=False):
        super().__init__()
        self.input_dim = input_dim
        if input_dim%heads:
            raise Exception
        self.heads = heads
        self.head_dim = input_dim//heads
        self.Q = nn.Linear(input_dim,input_dim)
        self.K = nn.Linear(input_dim,input_dim)
        self.V = nn.Linear(input_dim,input_dim)
        self.temperature = 1.0
        self.softmax = Softmax(dim=-1)
        self.layernorm = torch.nn.LayerNorm(self.head_dim)
        self.dropout = Dropout(drop_rate)
        self.is_decoder = is_decoder


    def forward(self,q,seq_len):
        #seq_len为输入语句的长度，用于softmax时mask掉信息 kv_seq_len用于cross attention时mask信息
        batch_size = q.shape[0]
        heads = self.heads
        q = self.Q(q)
        #print(kv.shape)
        k = self.K(q)
        v = self.V(q)
        #1.把每个词拆分成多个头 2.变换头的维度进去 每个头放外面，序列长度维度进去
        q = torch.transpose(q.view(batch_size,-1,heads,self.head_dim),-2,-3)
        k = torch.transpose(k.view(batch_size,-1,heads,self.head_dim),-2,-3)
        v = torch.transpose(v.view(batch_size,-1,heads,self.head_dim),-2,-3)
        q = self.layernorm(q)
        k = self.layernorm(k)
        v = self.layernorm(v)

        k = torch.transpose(k,-1,-2)
        attn = torch.matmul(q,k).view(-1,q.shape[-2],k.shape[-1])
        if self.is_decoder:
            mask_attn = decoder_mask(seq_len=seq_len)
            mask_attn = mask_attn.unsqueeze(1).repeat(1,  self.heads,1, 1)

            mask_attn = mask_attn.view(mask_attn.shape[0]*mask_attn.shape[1],-1,mask_attn.shape[-1])
            #print("decoder_shape,", mask_attn.shape, attn.shape)
            attn = attn * mask_attn
            attn = self.softmax(attn)
            attn = attn.view( -1, q.shape[-2], k.shape[-1])

        else:
            mask_attn = encoder_mask(seq_len)
            mask_attn = mask_attn.unsqueeze(1).repeat(1, self.heads,1, 1)
            mask_attn = mask_attn.view(attn.shape)
            attn = attn * mask_attn
            attn = attn.view( -1, q.shape[-2], k.shape[-1])
            attn = self.softmax(attn)


        v = v.view(-1,q.shape[-2],q.shape[-1])
        attn = self.dropout(attn)
        #print(attn.shape,v.shape)
        x = torch.matmul(attn,v)
        x = x.view(batch_size,-1,self.input_dim)
        return x


class MultiheadCrossAttention(nn.Module):
    def __init__(self,input_dim,heads,drop_rate=0.5):
        super().__init__()
        self.input_dim = input_dim
        if input_dim%heads:
            raise Exception
        self.heads = heads
        self.head_dim = input_dim//heads
        self.Q = nn.Linear(input_dim,input_dim)
        self.K = nn.Linear(input_dim,input_dim)
        self.V = nn.Linear(input_dim,input_dim)
        self.temperature = 1.0
        self.softmax = Softmax(dim=-1)
        self.layernorm = torch.nn.LayerNorm(self.head_dim)
        self.dropout = Dropout(drop_rate)


    def forward(self,q,seq_len,kv=None,kv_seq_len=None):
        #seq_len为输入语句的长度，用于softmax时mask掉信息 kv_seq_len用于cross attention时mask信息
        batch_size = q.shape[0]
        heads = self.heads
        q = self.Q(q)
        #print(kv.shape)
        k = self.K(kv)
        v = self.V(kv)
        #1.把每个词拆分成多个头 2.变换头的维度进去 每个头放外面，序列长度维度进去
        q = torch.transpose(q.view(batch_size,-1,heads,self.head_dim),-2,-3)
        k = torch.transpose(k.view(batch_size,-1,heads,self.head_dim),-2,-3)
        v = torch.transpose(v.view(batch_size,-1,heads,self.head_dim),-2,-3)
        q = self.layernorm(q)
        k = self.layernorm(k)
        v = self.layernorm(v)

        k = torch.transpose(k,-1,-2)
        attn = torch.matmul(q,k).view(-1,q.shape[-2],k.shape[-1])

        mask_attn = cross_mask(q_len=seq_len,kv_len=kv_seq_len)
        #print(mask_attn)
        mask_attn = mask_attn.unsqueeze(1).repeat(1,  self.heads,1, 1)

        mask_attn = mask_attn.view(mask_attn.shape[0]*mask_attn.shape[1],-1,mask_attn.shape[-1])
        #print("decoder_shape,", mask_attn.shape, attn.shape)
        attn = attn * mask_attn
        attn = self.softmax(attn)
        attn = attn.view( -1, q.shape[-2], k.shape[-1])



        v = v.view(-1,k.shape[-1],k.shape[-2])
        attn = self.dropout(attn)
        #print(attn.shape,v.shape)
        x = torch.matmul(attn,v)
        x = x.view(batch_size,-1,self.input_dim)
        return x



if __name__=='__main__':
    atten = MultiheadCrossAttention(768,4)
    x = torch.randn([3,4,768])
    kv = torch.randn([3,3,768])
    x = atten(x,kv=kv,seq_len = torch.tensor([1,2,4]),
              kv_seq_len = torch.tensor([3,2,3]))

    print(x.shape)

