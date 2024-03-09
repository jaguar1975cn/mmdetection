_base_ = '../../projects/CO-DETR/configs/codino/co_dino_5scale_r50_8xb2_1x_coco.py'

auto_scale_lr = dict(base_batch_size=8)

max_epochs = 40
train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=1)
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'))

def override_dict_functional(original_dict, key_to_override, new_value):
    return dict(map(lambda item: (key_to_override, new_value) if item[0] == key_to_override else item, original_dict.items()))

num_classes = 3

bbox_head=override_dict_functional(_base_.model['roi_head'][0]['bbox_head'], 'num_classes', num_classes)

model = dict(
    bbox_head=[override_dict_functional(_base_.model.bbox_head[0], 'num_classes', num_classes)],
    query_head=dict(num_classes=num_classes),
    roi_head=[
        override_dict_functional(_base_.model.roi_head[0], 'bbox_head', bbox_head)
    ],
)

data_root = './datasets/pklot/images/'
metainfo = {
    'classes': ('spaces', 'space-empty', 'space-occupied'),
    'palette': [
        (220, 220, 220),
        (20, 220, 60),
        (220, 20, 60),
    ]
}
annotation_file = '_annotations.coco-seg.json'
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/' + annotation_file,
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/' + annotation_file,
        data_prefix=dict(img='valid/')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test/' + annotation_file,
        data_prefix=dict(img='test/')))

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'valid/' + annotation_file, metric=['bbox'])
test_evaluator = dict(ann_file=data_root + 'test/' + annotation_file, metric=['bbox'])
