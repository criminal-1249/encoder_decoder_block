import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product(q, k, v, mask = None):
    d_k = k.shape[-1]
    scaled = torch.matmul(q, k.transpose(-1,-2))/math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim = -1)
    values = torch.matmul(attention,v) 

    return values,attention

class MultiHeadAttention(nn.Module):

    def __init__(self,d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model//num_heads

        self.qkv_layer = nn.Linear(d_model, 3*d_model)
        self.linear = nn.Linear(d_model,d_model)
        
    def forward(self,input,mask = None):
        batch_size, seq_len, d_model = input.shape # 30,200,512
        qkv = self.qkv_layer(input) #30,200,3*512
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3*self.head_dim)
        ## 30,200,8,3*512//8
        qkv = qkv.permute(0,2,1,3) #30,8,200,3*x
        q, k, v = qkv.chunk(3,dim = -1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0,2,1,3)
        values = values.reshape(batch_size, seq_len, self.d_model)
        
        out = self.linear(values)
        return out
    
class LayerNormalization(nn.Module):

    def __init__(self, parameter_shape, eps = 1e-5):
        super().__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameter_shape))
        self.beta = nn.Parameter(torch.zeros(parameter_shape))
    
    def forward(self, inputs):
        mean_ip = inputs.mean(dim = -1, keepdim = True)
        var_ip = inputs.var(dim = -1, keepdim = True, unbiased = True)

        y = (inputs-mean_ip)/torch.sqrt(var_ip+self.eps)
        out = self.gamma*y + self.beta

        return out

class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob):
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
        
class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model = d_model,num_heads = num_heads)
        self.norm1 = LayerNormalization(parameter_shape = [d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionWiseFeedForward(d_model = d_model,hidden = ffn_hidden,drop_prob = drop_prob)
        self.norm2 = LayerNormalization(parameter_shape = [d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        residual_x = x
        x = self.attention(x, mask = None)
        x = self.dropout1(x)
        x = self.norm1(x+residual_x)
        residual_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x+residual_x)

        return x

class Encoder(nn.Module):

    def __init__(self, d_model, num_heads, ffn_hidden, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])
        
    def forward(self, x):
        x = self.layers(x)
        return x 