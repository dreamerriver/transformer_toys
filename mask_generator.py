import torch


def encoder_mask(seq_len):
    #构造 [batch_size]
    x = torch.arange(0,max(seq_len)).unsqueeze(0).repeat(len(seq_len),1)

    seq_len =  seq_len.unsqueeze(1).repeat(1,len(x[1]))

    seq_len = seq_len.unsqueeze(1)

    seq_len = seq_len.repeat(1,seq_len.shape[2],1)
    #print(seq_len)
    x = x.unsqueeze(1).repeat(1,seq_len.shape[2],1)
    x = x < seq_len
    return x

def decoder_mask(seq_len):
    x = encoder_mask(seq_len)
    #此外额外处理mask, [batch_size,seq_len,seq_len] 对最后一个维度取值 1，2，3，……seq_len 在 长度位置被截断
    mask = torch.arange(0, max(seq_len)).unsqueeze(0).repeat(len(seq_len), 1)
    mask = mask.unsqueeze(1).repeat(1, len(mask[2]),1)
    #print(mask)
    #print("qlen",seq_len)
    mask_limit = torch.arange(0, max(seq_len)).unsqueeze(0).repeat(len(seq_len), 1)
    mask_limit = mask_limit.unsqueeze(2).repeat(1,1,len(mask[2]))
    mask = mask<=mask_limit
    x = x*mask
    return x

def cross_mask(q_len,kv_len):
    kv_mask = torch.arange(0, max(kv_len)).unsqueeze(0).repeat(len(q_len), 1).unsqueeze(1).repeat(1,max(q_len),1)
    kv_mask_limit = kv_len.unsqueeze(1).repeat(1,kv_mask.shape[-1]).unsqueeze(1).repeat(1,max(q_len),1)
    kv_mask = kv_mask < kv_mask_limit
    return kv_mask



if __name__ == '__main__':
    #y = encoder_mask(torch.tensor([2,2,4]))
    #y = decoder_mask(torch.tensor([4,5,6,7,10]))
    y = cross_mask(torch.tensor([3,4,5]),torch.tensor([4,5,6]))
    print(y)