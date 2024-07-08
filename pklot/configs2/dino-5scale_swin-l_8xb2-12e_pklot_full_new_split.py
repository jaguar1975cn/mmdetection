_base_ = '../../configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py'
auto_scale_lr = dict(base_batch_size=8)

max_epochs = 20
num_levels = 4
train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=1)
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'))

model = dict(
        num_feature_levels=num_levels,
        num_queries=800,
    bbox_head=dict(num_classes=3),
    neck=dict(in_channels=[192, 384, 768, 1536], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels)))
    )

data_root = './datasets/pklot/pklot_full_new_split/'
metainfo = {
    'classes': ('spaces', 'space-empty', 'space-occupied'),
    'palette': [
        (220, 220, 220),
        (20, 220, 60),
        (220, 20, 60),
    ]
}
annotation_file = 'pklot_ext_annotations.json'
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/' + annotation_file,
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/' + 'pklot_ext_annotations-1024.json',
        data_prefix=dict(img='valid/')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test/' + annotation_file,
        data_prefix=dict(img='test/')))

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'valid/' + annotation_file, metric=['bbox'], proposal_nums=[1000, 1000, 1000], use_mp_eval=True)
test_evaluator = dict(ann_file=data_root + 'test/' + annotation_file, metric=['bbox'], proposal_nums=[1000, 1000, 1000], use_mp_eval=True)
