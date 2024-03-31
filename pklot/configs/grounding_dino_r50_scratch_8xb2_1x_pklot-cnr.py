_base_ = '../../configs/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_coco.py'
#auto_scale_lr = dict(base_batch_size=8)

max_epochs = 40
train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=1)
default_hooks = dict(
    checkpoint=dict(interval=5, type='CheckpointHook'))

model = dict(
    bbox_head=dict(num_classes=3))

#data_root = './datasets/pklot/images/'
data_root = './datasets/cnr/'
metainfo = {
    'classes': ('spaces', 'space-empty', 'space-occupied'),
    'palette': [
        (220, 220, 220),
        (20, 220, 60),
        (220, 20, 60),
    ]
}
#annotation_file = '_annotations.coco-seg.json'
annotation_file = 'cnr-coco.json'
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
        ann_file= annotation_file,
        data_prefix=dict(img='')))

# Modify metric related settings
#val_evaluator = dict(ann_file=data_root + 'valid/' + annotation_file, metric=['bbox'])
test_evaluator = dict(ann_file=data_root + annotation_file, metric=['bbox'])
