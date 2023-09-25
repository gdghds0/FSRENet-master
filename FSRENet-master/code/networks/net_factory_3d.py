
from networks.vnet import VNet
from networks.vnet import VNet

def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2):
    
    if net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()

    else:
        net = None
    return net
