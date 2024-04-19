import argparse
from configs.defaults import _C as cfg_default

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/datasets", help="path to dataset")
    parser.add_argument("--algorithm", type=str, default='DG_ADR', help='check in algorithms.py')
    parser.add_argument("--desc", type=str, default="name_of_exp")
    parser.add_argument("--backbone", type=str, default="resnet50")
    #parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DGDR")
    #parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DGDR")
    #parser.add_argument("--random", action="store_true") 
    parser.add_argument("--dg_mode", type=str, default='DG', help="DG or ESDG")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_epochs", type=int, default=5)
    parser.add_argument("--output", type=str, default='test')
    parser.add_argument("--override", action="store_true") 

    #more args
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optim", type=str, default='sgd')
    parser.add_argument("--timestamp", type=str, default='not_resuming', help="useful for resuming training")
    parser.add_argument("--project_name", type=str, default='DG_in_DR_seed_0', help="Main project name")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--target", type=str, default='deepdr')
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--sd_param", type=float, default=0.0001) # for the SD method
    #initilization weights
    parser.add_argument("--ssl_pretrained", action="store_true")
    parser.add_argument("--imagenet_pretrained", action="store_true") 
    parser.add_argument("--checkpoint_path", type=str, default='/', help="Path to SSL checkpoint") 
    # DG_ADR params
    parser.add_argument("--trivial_aug", action="store_true") # augmentation strategy
    parser.add_argument("--warm_up_epochs", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--margin", type=float, default=0.001)
    parser.add_argument("--loss_alpha", type=float, default=1.0)  # weighting param for domain alignment loss
    parser.add_argument("--weight_loss_alpha", type=float, default=1.0) # wighting param for focal loss
    parser.add_argument("--use_syn", action="store_true") # use sythentic data as augmentations
    return parser.parse_args()

def setup_cfg(args):
    cfg = cfg_default.clone()
    #cfg.RANDOM = args.random
    #cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    #cfg.DATASET.TARGET_DOMAINS = args.target_domains
    cfg.SEED = args.seed
    cfg.OUTPUT_PATH = args.output
    cfg.OVERRIDE = args.override
    cfg.DG_MODE = args.dg_mode

    cfg.DESCRIPTION = args.desc
    cfg.ALGORITHM = args.algorithm
    cfg.BACKBONE = args.backbone
    
    cfg.DATASET.ROOT = args.root
    cfg.DATASET.NUM_CLASSES = args.num_classes
    cfg.VAL_EPOCH = args.val_epochs

    cfg.OPTIMIZER = args.optim
    cfg.EPOCHS = args.num_epochs
    cfg.LEARNING_RATE = args.lr
    cfg.BATCH_SIZE = args.batch_size
    cfg.VAL_BATCH_SIZE = args.val_batch_size
    cfg.WEIGHT_DECAY = args.weight_decay
    cfg.MOMENTUM = args.momentum
    cfg.DROP_OUT = args.dropout
    cfg.TIMESTAMP = args.timestamp
    cfg.PROJECT_NAME = args.project_name
    cfg.TARGET = args.target
    cfg.DEBUG = args.debug
    cfg.SD_PARAM = args.sd_param

    cfg.SSL_PRETRAINED = args.ssl_pretrained
    cfg.IMAGENET_PRETRAINED = args.imagenet_pretrained
    cfg.CHECKPOINT_PATH = args.checkpoint_path
    cfg.TRIVIAL_AUG = args.trivial_aug
    cfg.DROPOUT = args.dropout
    cfg.WARM_UP_EPOCHS = args.warm_up_epochs
    cfg.MARGIN = args.margin
    cfg.K = args.k
    cfg.LOSS_ALPHA = args.loss_alpha
    cfg.USE_SYN = args.use_syn
    cfg.WEIGHT_LOSS_ALPHA = args.weight_loss_alpha

    if args.dg_mode == 'DG':
        cfg.merge_from_file("./configs/datasets/GDRBench.yaml")
    elif args.dg_mode == 'ESDG':
        cfg.merge_from_file("./configs/datasets/GDRBench_ESDG.yaml")
    else:
        raise ValueError('Wrong type')

    return cfg

