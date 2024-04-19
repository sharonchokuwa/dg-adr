
from .resnet import resnet18, resnet50, resnet101
from .nets import *
import torch


def get_net(cfg):
    if cfg.ALGORITHM == 'ERM' or cfg.ALGORITHM == 'GDRNet' or cfg.ALGORITHM == 'DG_ADR' or cfg.ALGORITHM == 'VAE_DG' or cfg.ALGORITHM == 'SelfReg' or cfg.ALGORITHM == 'SD':
        net = get_backbone(cfg)
    elif cfg.ALGORITHM == 'MixupNet':
        net = MixupNet(cfg)
    elif cfg.ALGORITHM == 'Fishr' or cfg.ALGORITHM == 'DRGen':
        net = FishrNet(cfg)
    else:
        raise ValueError('Wrong type')
    return net


def get_backbone(cfg):
    if cfg.BACKBONE == 'resnet18':
        model = resnet18(pretrained=True)
    elif cfg.BACKBONE == 'resnet50':
        if cfg.SSL_PRETRAINED:
            model = resnet50(ssl_pretrained=True, checkpoint_path=cfg.CHECKPOINT_PATH, dropout_rate=cfg.DROPOUT)
        elif cfg.IMAGENET_PRETRAINED:
            model = resnet50(imagenet_pretrained=True)
        else:
            model = resnet50()
    elif cfg.BACKBONE == 'resnet101':
        model = resnet101(pretrained=True)
    else:
        raise ValueError('Wrong type')
    return model

def get_classifier(out_feature_size, cfg):
    return torch.nn.Linear(out_feature_size, cfg.DATASET.NUM_CLASSES)