import torch
from torch import nn
from multihead_selfattention import MultiheadSelfAttention,MultiheadCrossAttention

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attlayer = MultiheadSelfAttention(input_dim=768,heads=12,drop_rate=0.5,is_decoder=False)
        self.linear = nn.Linear(768,768)
        self.layernorm = nn.LayerNorm(768)

    def forward(self,x,seq_len):
        #shortcut层
        x = x + self.attlayer(x,seq_len)
        x = self.layernorm(self.linear(x)) + x
        return x

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attlayer1 = MultiheadSelfAttention(input_dim=768,heads=12,drop_rate=0.5,is_decoder=True)
        self.attlayer2 = MultiheadCrossAttention(input_dim=768,heads=12,drop_rate=0.5)
        self.linear = nn.Linear(768,768)
        self.layernorm = nn.LayerNorm(768)

    def forward(self,q,kv,seq_len,kv_seq_len):
        #kv来自encoder
        #q, seq_len, kv = None, kv_seq_len = None
        #print(q.shape,self.attlayer1(q=q,kv=kv,seq_len=seq_len,kv_seq_len=max(seq_len)).shape)
        x = q + self.attlayer1(q=q,seq_len=seq_len)
        x = x + self.attlayer2(q=q,kv=kv,seq_len=seq_len,kv_seq_len=kv_seq_len)
        x = self.layernorm(self.linear(x)) +x
        return x

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([EncoderLayer() for _ in range(12)])

    def forward(self,x,seq_len):
        for m in self.encoders:
            x = m(x,seq_len)
        return x
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoders = nn.ModuleList([DecoderLayer() for _ in range(12)])

    def forward(self,x,kv,seq_len,kv_seq_len):
        for m in self.decoders:
            x = m(x,kv,seq_len,kv_seq_len)
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self,x_encoder,x_decoder,seq_len_encoder,seq_len_decoder):
        encoder_out = self.encoder(x_encoder,seq_len_encoder)
        decoder_out = self.decoder(x_decoder,encoder_out,seq_len_decoder,seq_len_encoder)
        return encoder_out,decoder_out


if __name__ == "__main__":
    x_encoder = torch.randn([8, 128, 768])
    #encoder_layer = EncoderLayer()
    #x = encoder_layer(x,torch.tensor([(i+8*3) for i in range(7)] +[128])).shape
    #decoder_layer = DecoderLayer()
    #encoder = Encoder()
    #kv = torch.randn([8, 128, 768])
    encoder_seq_len = torch.tensor([(i + 8 * 3) for i in range(7)] + [128])
    decoder_seq_len = torch.tensor([(i + 8 * 3) for i in range(7)] + [70])
    #print(x.shape,kv.shape)
    #x = decoder_layer(q=x,seq_len=seq_len,kv=kv,kv_seq_len=kv_seq_len)
    #kv = encoder(x,kv_seq_len)
    x_decoder = torch.randn([8, 70, 768])
    transformer = Transformer()
    encoder_out,decoder_out = transformer(x_encoder,x_decoder,encoder_seq_len,decoder_seq_len)
    #decoder = Decoder()
    #x = decoder(x,kv,seq_len,kv_seq_len)
    print(decoder_out)
