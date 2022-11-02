import torch
import torch.nn as nn

from models.position_encode import PositionEncoding2D
from models.transformers import TransFormerBlock


class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """
    def __init__(self, in_channels, out_channels, 
        kernel_size, stride=1, padding=0, 
        pre_norm=None, post_norm=nn.BatchNorm2d, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.post_norm(x) # [B, C, H, W]
        # x = x.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        
        return x
    
    
class InterleavingAttention(nn.Module):
    def __init__(self, dim, emb_dim, modes):
        super().__init__()
        self.modes = modes
        self.ds1 = Downsampling(dim, emb_dim, kernel_size=7, stride=2, padding=7//2)
        self.ds2 = Downsampling(dim, emb_dim, kernel_size=7, stride=2, padding=7//2)
        self.pe = PositionEncoding2D(emb_dim)
        self.transformers = nn.ModuleList([TransFormerBlock(emb_dim, mode=modes[i]) for i in range(len(modes))])
        

    def forward(self, x1, x2):
        x1, x2 = self.ds1(x1), self.ds2(x2)        
        pe1, pe2 = self.pe(x1), self.pe(x2)
        x1, x2 = x1 + pe1, x2 + pe2
        # [B, C, H, W] -> [B, H, W, C]
        x1 = x1.permute(0, 2, 3, 1) 
        x2 = x2.permute(0, 2, 3, 1)
        
        for i in range(0, len(self.transformers), 2):
            print(i)
            if self.modes[i] == 'self':
                delta1 = self.transformers[i](x1, x1)
                delta2 = self.transformers[i+1](x2, x2)
                x1, x2 = x1 + delta1, x2 + delta2
            else:
                delta1 = self.transformers[i](x1, x2)
                delta2 = self.transformers[i+1](x2, x1)
                x1, x2 = x1 + delta1, x2 + delta2

            
        return x1, x2