
from mmdet.registry import MODELS
from mmdet.models.dense_heads.ssd_head import SSDHead

@MODELS.register_module()
class MySSDHead(SSDHead):
    def __init__(self, *args, **kwargs):
        super(MySSDHead, self).__init__(*args, **kwargs)
        self.loss_cls = {}