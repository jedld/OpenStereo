# @Time    : 2024/9/21 11:39
# @Author  : josephemmanueldayo
from stereo.modeling.trainer_template import TrainerTemplate
from .usamnet import USAMNet, USAMNetv2
from .usamnet import UNet, SaUSAMNet, SaUNet

__all__ = {
    'USAMNet': USAMNet,
    'UNet': UNet,
    'SaUSAMNet': SaUSAMNet,
    'SaUNet': SaUNet,
    'USAMNetv2': USAMNetv2
}

class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)
