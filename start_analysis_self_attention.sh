#!/bin/sh

# This script is used to start the analysis of the self-attention mechanism in the transformer model.

python tools/image_analysis.py --cfg_file=cfgs/usamnet/unet_kitti.yaml --cfg_file2=cfgs/usamnet/usamnet_kitti.yaml --pretrained_model=output/KittiDataset/UNet/unet_kitti/default/ckpt/checkpoint_epoch_1999.pth --pretrained_model2=output/KittiDataset/USAMNet/usamnet_kitti/default/ckpt/checkpoint_epoch_1999.pth --save_dir=output/KittiDataset/UNetVsUSAMNetAnalysis