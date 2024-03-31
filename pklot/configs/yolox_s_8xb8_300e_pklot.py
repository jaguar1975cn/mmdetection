_base_ = '../../configs/yolox/yolox_s_8xb8-300e_coco.py'

max_epochs = 40
train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=1)
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'))

model = dict(
    bbox_head=dict(num_classes=3))

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

train_dataset = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='train/' + annotation_file,
        data_prefix=dict(img='train/'),
        metainfo=metainfo
    )
)

train_dataloader = dict(
    # batch_size=1,
    dataset=train_dataset
)

val_dataloader = dict(
    # batch_size=1,
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
