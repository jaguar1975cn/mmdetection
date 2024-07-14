_base_ = '../../configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
auto_scale_lr = dict(base_batch_size=8)

max_epochs = 20
train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=1)
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'))

model = dict(
    bbox_head=dict(num_classes=3),
#    backbone=dict(depth=18),
    decoder=dict(num_layers=3),
    encoder=dict(num_layers=3),
    num_queries=300
)

data_root = './datasets/cnr/'
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
        ann_file='cnr-train.json',
        data_prefix=dict(img='')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='cnr-validation.json',
        data_prefix=dict(img='')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='cnr-test.json',
        data_prefix=dict(img='')))

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'cnr-validation.json', metric=['bbox'], proposal_nums=[300, 300, 300], use_mp_eval=True)
test_evaluator = dict(ann_file=data_root + 'cnr-test.json', metric=['bbox'], proposal_nums=[300, 300, 300], use_mp_eval=True)
