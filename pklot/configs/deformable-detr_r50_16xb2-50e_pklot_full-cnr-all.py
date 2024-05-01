_base_ = '../../configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'

auto_scale_lr = dict(base_batch_size=8)

max_epochs = 290
train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=1)
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'))

model = dict(
    bbox_head=dict(num_classes=3),
    test_cfg=dict(max_per_img=300)
    )

data_root = './datasets/'
metainfo = {
    'classes': ('spaces', 'space-empty', 'space-occupied'),
    'palette': [
        (220, 220, 220),
        (20, 220, 60),
        (220, 20, 60),
    ]
}
annotation_file = 'cnr-coco.json'
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='merged_train.json',
        data_prefix=dict(img='')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='merged_valid_1024.json',
        data_prefix=dict(img='')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='merged_test.json',
        data_prefix=dict(img='')))

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'merged_valid.json', metric=['bbox'])
test_evaluator = dict(ann_file=data_root + 'merged_test.json', metric=['bbox'])
