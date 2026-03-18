import torch
from .dpnet import DPNET

def DPMODEL(symbol_features: int, embedding_layers: list, Rcut: float, fitting_layers: list):
    """
    wgdpmodel
    """
    lnorm=True
    norm_type='layer' #'batch' 'layer'
    bias=False
    model = DPNET(symbol_features=symbol_features, embedding_layers=embedding_layers, Rcut=Rcut, fitting_layers=fitting_layers, lnorm=lnorm, norm_type=norm_type, bias=bias)
    return model
