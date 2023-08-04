import torch
import disentangle as dis
from recognition import *

pred = torch.rand(16, 100)
label = torch.ones(16, 100)

rec1 = Recognition(num_class=100)
rec1.reset()
rec1.update(label, pred)
res_i = rec1.compute_AP('i')
print('1')