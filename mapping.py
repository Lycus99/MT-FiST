import numpy as np
import torch

map_dict_url = './dict/maps.txt'
maps_dict    = np.genfromtxt(map_dict_url, dtype=int, comments='#', delimiter=',', skip_header=0)

IVT, I, V, T = [], [], [], []
with open('./dict/maps.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        if i==0: continue
        v1, v2, v3, v4 = line.strip().split(',')[:4]
        IVT.append(int(v1))
        I.append(int(v2))
        V.append(int(v3))
        T.append(int(v4))
IVT, I, V, T = np.array(IVT), np.array(I), np.array(V), np.array(T)

preds_4 = torch.zeros(10, 100)
for tool, verb, target in zip(I, V, T):
    # print(tool, verb, target)
    # tool_idx, verb_idx, target_idx = np.where(tool), np.where(verb), np.where(target)
    # if tool_idx and verb_idx and target_idx:
    select_index = np.where( 
        np.logical_and(np.logical_and(I==tool, V==verb), T==target))[0]
    if not len(select_index) == 0:
        print(select_index)
        preds_4[0][select_index] = 1