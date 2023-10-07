# from .CLNet import CLNet
# from .Transformer import TransformerNet
# from .ResNet import Resnet
from .MC_Loss import *

model_zoo = {
    # 'CLNet': CLNet,
    # 'TransformerNet':TransformerNet,
    # 'res':Resnet,
    'tool': model_tool,
    'verb':model_verb,
    'target':model_target,
    'triplet':model_triplet,
    'all':model_all,
    'all4':model_all4,
    'model_all4_cam':model_all4_cam,
    'mc_lstm': model_lstm,
    'model_lstm_cam': model_lstm_cam
}

def model_provider(name, **kwargs):

    model_ret = model_zoo[name](**kwargs)
    
    return model_ret
