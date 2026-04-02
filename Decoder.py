import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product(q, k, v, mask = None):
    d_k = k.shape[-1] # 30 8 200 64
    scaled = torch.matmul(q,k.transpose(-1,-2))/math.sqrt(d_k)
    print(f"scaled shape : {scaled.shape}")
    if mask is not None:
        
        scaled += mask # 30 8 200 200
    attention = F.softmax(scaled,dim = -1) 
    values = torch.matmul(attention, v) # 30 8 200 64

    return values, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_layer = nn.Linear(d_model, 3*d_model)
        self.linear = nn.Linear(d_model,d_model)
    
    def forward(self, x, mask = None):
        batch_size,seq_len,d_model = x.shape
        print(f"x shape = {x.shape}")
        qkv = self.qkv_layer(x)
        print(f"qkv shape : {qkv.shape}")
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0,2,1,3)
        q, k, v = qkv.chunk(3, dim = -1) # 30 8 200 64
        values,attention = scaled_dot_product(q,k,v,mask) # 30 8 200 64
        values = values.permute(0,2,1,3) # 30 200 8 64
        values = values.reshape(batch_size,seq_len,self.num_heads*self.head_dim)

        out = self.linear(values) # 30 200 512
        return out
    
class MultiHeadCrossAttention(nn.Module):
    def __init__(self,d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model//num_heads
        
        self.kv_layer = nn.Linear(d_model,2*d_model)
        self.q_layer = nn.Linear(d_model,d_model)
        self.linear_layer = nn.Linear(d_model,d_model)

    def forward(self, x, y, mask = None):
        batch_size,seq_len,d_model = x.shape # 30 200 512
        kv = self.kv_layer(x) # 30 200 2*512
        q = self.q_layer(y) # 30 200 512

        # 30 200 8 2*64
        kv = kv.reshape(batch_size,seq_len,self.num_heads,2*self.head_dim)
        q = q.reshape(batch_size,seq_len,self.num_heads,self.head_dim)

        kv = kv.permute(0,2,1,3) # 30 8 200 64
        q = q.permute(0,2,1,3)

        k,v = kv.chunk(2,dim = -1)
        values, attention = scaled_dot_product(q, k, v, mask)# 30 8 200 64

        values = values.permute(0,2,1,3) # 30 200 8 64
        # 30 200 8*64
        values = values.reshape(batch_size, seq_len, d_model)

        out = self.linear_layer(values)
        return out
    
class LayerNormalization(nn.Module):
    def __init__(self,parameters_shape, eps = 1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))
    
    def forward(self,inputs):
        mean_ip = inputs.mean(dim = -1,keepdim = True)
        var_ip = inputs.var(dim = -1, keepdim = True, unbiased = False)

        std = torch.sqrt(var_ip + self.eps)
        y = (inputs - mean_ip)/std
        out = self.gamma*y + self.beta

        return out

class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model,hidden)
        self.linear2 = nn.Linear(hidden,d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = drop_prob)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x
    
class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model = d_model, num_heads = num_heads)
        self.norm1 = LayerNormalization(parameters_shape = [d_model])
        self.dropout1 = nn.Dropout(p = drop_prob)
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model = d_model, num_heads = num_heads)
        self.norm2 = LayerNormalization(parameters_shape = [d_model])
        self.dropout2 = nn.Dropout(p = drop_prob)
        self.ffn = PositionWiseFeedForward(d_model,hidden = ffn_hidden, drop_prob = drop_prob)
        self.norm3 = LayerNormalization(parameters_shape = [d_model])
        self.dropout3 = nn.Dropout(p = drop_prob)
    
    def forward(self, x, y, decoder_mask):
        residual_y = y
        print("Masked self attention")
        y = self.self_attention(y, mask = decoder_mask)
        print("Drop out 1")
        y = self.dropout1(y)
        print("layer norm")
        y = self.norm1(y+residual_y)

        residual_y = y
        print("cross attention")
        y = self.encoder_decoder_attention(x, y, mask = None)
        print("Drop out 2")
        y = self.dropout2(y)
        y = self.norm2(y+residual_y)

        residual_y = y
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.norm3(y+residual_y)

        return y
    
class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, mask = inputs
        for module in self._modules.values():
            y = module(x, y, mask)
        return y
    
class Decoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])
    
    def forward(self, x, y, mask):
        # x,y -> 30 200 512
        y = self.layers(x,y,mask)
        return y
    