import sys
import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
from easydict import EasyDict
from PIL import Image

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.modeling import build_trainer
from stereo.datasets.dataset_template import build_transform_by_cfg

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for eval')
    parser.add_argument('--cfg_file2', type=str, default=None, help='specify the config2 for eval')
    parser.add_argument('--pretrained_model', type=str, default=None)

    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

    yaml_config2 = common_utils.config_loader(args.cfg_file2)
    cfgs2 = EasyDict(yaml_config2)

    args.run_mode = 'infer'
    return args, cfgs, cfgs2

def main():
    args, cfgs, cfgs2 = parse_config()
    if args.dist_mode:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
    else:
        local_rank = 0
        global_rank = 0

    # env
    torch.cuda.set_device(local_rank)
    seed = 0 if not args.dist_mode else dist.get_rank()
    common_utils.set_random_seed(seed=seed)

    # log
    logger = common_utils.create_logger(log_file=None, rank=local_rank)

    # log args and cfgs
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_configs(cfgs, logger=logger)

    # model
    trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, None)
    model = trainer.model

    trainer2 = build_trainer(args, cfgs2, local_rank, global_rank, logger, None)
    model2 = trainer2.model

    print(model)
    print(model2)

if __name__ == '__main__':
    main()

