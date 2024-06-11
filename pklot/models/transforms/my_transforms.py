#from mmdet.datasets.pipelines.builder import PIPELINES
from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
import random
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type

@TRANSFORMS.register_module()
class RandomBboxScaler(BaseTransform):
    def __init__(self, scale_range=(0.8, 1.2), scale_prob=0.5):
        self.scale_range = scale_range
        self.scale_prob = scale_prob

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        if random.random() < self.scale_prob:
            scale = (random.uniform(*self.scale_range), random.uniform(*self.scale_range))
            for key in results.get('bbox_fields', ['gt_bboxes']):
                results[key].rescale_(scale)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_range={self.scale_range}, scale_prob={self.scale_prob})'