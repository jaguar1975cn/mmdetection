from mmdet.registry import MODELS
from mmdet.models import DetDataPreprocessor
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union
import json
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

@MODELS.register_module()
class MaskedDataPreprocessor(DetDataPreprocessor):
    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 mask_polygon: List[Tuple[int, int]] = None,
                 masked_images_annotation_file: str = None, # annotation file
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            mask_pad_value=mask_pad_value,
            pad_seg=pad_seg,
            seg_pad_value=seg_pad_value,
            boxtype2tensor=boxtype2tensor,
            non_blocking=non_blocking,
            batch_augments=batch_augments
            )
        print('MaskedDataPreprocessor is used.')
        # create a closed polygon
        self.mask_polygon = mask_polygon + [mask_polygon[0]]
        self.mask = None
        self.masked_images_annotation_file = masked_images_annotation_file
        self.masked_images = set()

        with open(masked_images_annotation_file, 'r') as f:
            data = json.load(f)
            for img in data['images']:
                self.masked_images.add(img['file_name'])

    def forward(self, data: dict, training: bool = False) -> dict:
        # print("shape of data", data['inputs'][0].shape)
        if self.mask is None:
            self.mask = self.create_polygon_mask(data['inputs'][0].shape[1:], self.mask_polygon)

        img = data['inputs'][0]

        img_path = data['data_samples'][0].img_path
        print("img_path:", img_path)
        # get file name
        file_name = img_path.split('/')[-1]
        # print("file_name", file_name)

        if file_name in self.masked_images:
            data['inputs'][0] = data['inputs'][0] * self.mask
            print("masked")

        img = data['inputs'][0]

        return super().forward(data, training)

    def create_polygon_mask(self, image_shape, polygon_vertices):
        # create a blank image with the same size as the input image
        image_mask = torch.zeros(image_shape, dtype=torch.uint8)

        # Convert polygon vertices to a tuple of tuples
        polygon_tuple = tuple(map(tuple, polygon_vertices))

        # Create a PIL Image
        pil_image = T.ToPILImage()(image_mask)

        # Create a draw object
        draw = ImageDraw.Draw(pil_image)

        # Fill the polygon with white color
        draw.polygon(polygon_tuple, outline=255, fill=255)

        # convert back to tensor
        image_mask_filled = T.ToTensor()(pil_image)

        # create binary mask
        image_mask_filled = (image_mask_filled > 0).to(torch.uint8)

        return image_mask_filled
