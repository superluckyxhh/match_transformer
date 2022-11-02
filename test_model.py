import torch
from models.transformers import TransFormerBlock
from models.backbone import InterleavingAttention

inp = torch.randn((10, 60, 80, 256))
# model = TransFormerBlock(256, mode='self')
# out = model(inp, inp)
# print(out.shape)

ims1 = torch.randn((10, 3, 640, 480)).cuda()
ims2 = torch.randn((10, 3, 640, 480)).cuda()

model = InterleavingAttention(3, 128, ['self', 'self'] * 2 + ['cross']*2).cuda()
out = model(ims1, ims2)
print()