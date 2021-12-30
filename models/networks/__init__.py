from .UNaah import *
from .UNaah_Nested import *

def get_network(name, backbone_name, n_classes=2, pretrained=True, encoder_freeze=False,
                shortcut_features='default', decoder_filter_block=64,decoder_use_batchnorm=True):


    if name == 'ParallelUnet':
        model = ParallelUnet(backbone_name=backbone_name,
                 pretrained=pretrained,
                 encoder_freeze=encoder_freeze,
                 shortcut_features=shortcut_features,
                 classes=n_classes,
                 decoder_filter_block=decoder_filter_block,
                 decoder_use_batchnorm=decoder_use_batchnorm)
    elif name == 'ParallelNestedUnet':
        model = ParallelNestedUnet(backbone_name=backbone_name,
                 pretrained=pretrained,
                 encoder_freeze=encoder_freeze,
                 shortcut_features=shortcut_features,
                 classes=n_classes,
                 decoder_filter_block=64, # has to be 64
                 decoder_use_batchnorm=decoder_use_batchnorm)

    else:
        raise 'Model {} is not available'.format(name)

    return model
