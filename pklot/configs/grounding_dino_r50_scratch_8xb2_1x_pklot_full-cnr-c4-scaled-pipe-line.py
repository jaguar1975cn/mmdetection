_base_ = '../../configs/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_coco.py'
#auto_scale_lr = dict(base_batch_size=8)

max_epochs = 40
train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=1)
default_hooks = dict(
    checkpoint=dict(interval=5, type='CheckpointHook'))

model = dict(
    bbox_head=dict(num_classes=3))

# data_root = './datasets/CNR-PARK/CNR-EXT_FULL_IMAGE_1000x750/'
data_root = './datasets/cnr/'
metainfo = {
    'classes': ('spaces', 'space-empty', 'space-occupied'),
    'palette': [
        (220, 220, 220),
        (20, 220, 60),
        (220, 20, 60),
    ]
}
annotation_file = 'cnr-coco-c4.json'
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
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        800,
        1333,
    ), type='FixScaleResize'),
    dict(keep_ratio=True, scale=(
        800/2,
        1333/2,
    ), type='FixScaleResize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'text',
            'custom_entities',
        ),
        type='PackDetInputs'),
]

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file= annotation_file,
        pipeline=test_pipeline,
        data_prefix=dict(img='')))

# Modify metric related settings
# val_evaluator = dict(ann_file=data_root + 'valid/' + annotation_file, metric=['bbox'])
test_evaluator = dict(ann_file=data_root + annotation_file, metric=['bbox'])
