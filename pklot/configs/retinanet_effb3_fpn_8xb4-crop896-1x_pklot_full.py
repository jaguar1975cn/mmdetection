_base_ = '../../configs/efficientnet/retinanet_effb3_fpn_8xb4-crop896-1x_coco.py'

max_epochs = 40
train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=1)

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
annotation_file = 'full3.json'
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        _delete_=True,
        type={{_base_.dataset_type}},
        metainfo=metainfo,
        data_root=data_root,
        ann_file='train/' + annotation_file,
        data_prefix=dict(img='train/'),
        pipeline=_base_.train_pipeline
    )
)
val_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type={{_base_.dataset_type}},
        metainfo=metainfo,
        data_root=data_root,
        ann_file='valid/' + annotation_file,
        data_prefix=dict(img='valid/'),
        pipeline=_base_.test_pipeline
    )
)

test_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type={{_base_.dataset_type}},
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test/' + annotation_file,
        data_prefix=dict(img='test/'),
        pipeline=_base_.test_pipeline
    )
)

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'valid/' + annotation_file, metric=['bbox'])
test_evaluator = dict(ann_file=data_root + 'test/' + annotation_file, metric=['bbox'])
