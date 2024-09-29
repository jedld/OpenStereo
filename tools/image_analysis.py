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
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--pretrained_model2', type=str, default=None)
    parser.add_argument('--pin_memory', action='store_true', default=False, help='data loader pin memory')
    parser.add_argument('--save_dir', type=str, default='./output', help='save root dir for this experiment')


    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)

    cfgs = EasyDict(yaml_config)
    if args.pretrained_model:
        cfgs.MODEL.PRETRAINED_MODEL = args.pretrained_model

    yaml_config2 = common_utils.config_loader(args.cfg_file2)

    cfgs2 = EasyDict(yaml_config2)
    if args.pretrained_model:
        cfgs2.MODEL.PRETRAINED_MODEL = args.pretrained_model2

    args.run_mode = 'eval'
    return args, cfgs, cfgs2, args.save_dir

def d1_metric(disp_pred, disp_gt, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0).to(disp_pred.device)
    disp_pred, disp_gt = disp_pred[mask], disp_gt[mask]
    E = torch.abs(disp_gt - disp_pred)
    err_mask = (E > 3) & (E / torch.abs(disp_gt) > 0.05)
    return torch.mean(err_mask.float()) * 100

def main():
    args, cfgs, cfgs2, save_dir = parse_config()

    local_rank = 0
    global_rank = 0

    # env
    torch.cuda.set_device(local_rank)
    seed = 1337
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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    difference_file = open(os.path.join(save_dir, 'difference.txt'), 'w')
    disparity_output_folder = os.path.join(save_dir, 'disparities')
    if not os.path.exists(disparity_output_folder):
        os.makedirs(disparity_output_folder)

    for data, disp_preds, disp_pred2s, disp_gts, masks in trainer.comparative_eval(model2):
        for filename, disp_pred, disp_pred2, disp_gt, mask in zip(data['name'], disp_preds, disp_pred2s, disp_gts, masks):

            metric_1 = d1_metric(disp_pred.squeeze(0), disp_gt, mask)
            metric_2 = d1_metric(disp_pred2.squeeze(0), disp_gt, mask)
            print(f"{filename}: {metric_1} - {metric_2} = {metric_1 - metric_2}")
            performance_difference = metric_1 - metric_2

            difference_file.write(f"{filename},{performance_difference}\n") 

            disp_pred = disp_pred.squeeze().cpu().numpy()
            disp_pred2 = disp_pred2.squeeze().cpu().numpy()
            # write disparities to a png file
            disp_pred = (disp_pred * 256).astype(np.uint16)
            disp_pred2 = (disp_pred2 * 256).astype(np.uint16)
            filename = os.path.basename(filename)[:-4] + '.png'
            Image.fromarray(disp_pred).save(os.path.join(disparity_output_folder, filename))
            Image.fromarray(disp_pred2).save(os.path.join(disparity_output_folder, '2_' + filename))



    difference_file.close()
if __name__ == '__main__':
    main()

