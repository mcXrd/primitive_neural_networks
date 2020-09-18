import torch
from torch import nn
import pprint

m = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=6)
input = torch.randn(7, 2, 6)
pprint.pprint(input)
pprint.pprint(input.shape)
pprint.pprint('xxx')
output = m(input)
pprint.pprint(output)
pprint.pprint(output.shape)
