import torch
import torch.nn as nn
import math

class PositionEncoding1D(nn.Module):
    def __init__(self, dim, max_len=1200):
        super().__init__()
        self.P = torch.zeros((1, max_len, dim))
        x = torch.arange(max_len) / torch.power(10000, torch.arange(0, dim, 2) / dim)
        self.P[:, :, 0::2] = torch.sin(x)
        self.P[:, :, 1::2] = torch.cos(x)
        
    def forward(self, x):
        pe = self.P[:, x.size(1), :]
        return pe
    
    
class PositionEncoding2D(nn.Module):
    def __init__(self, d_model, max_shape=(640, 480), temperature=10000.):
        super().__init__()
        dim = d_model // 2

        pe = torch.zeros((d_model, *max_shape))
        pe.requires_grad = False

        y_pos = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_pos = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)

        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(temperature)/dim))
        div_term = div_term[:, None, None] #[64, 1, 1]
        # x0, y0, x1, y1 -> 4
        pe[0::4, :, :] = torch.sin(x_pos * div_term)
        pe[1::4, :, :] = torch.cos(x_pos * div_term)
        pe[2::4, :, :] = torch.sin(y_pos * div_term)
        pe[3::4, :, :] = torch.cos(y_pos * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor):
        return self.pe[:, :, :x.size(2), :x.size(3)] 
        