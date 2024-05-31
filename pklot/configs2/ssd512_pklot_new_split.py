_base_ = '../../configs/ssd/ssd512_coco.py'

custom_imports = dict(
    imports=['pklot'],
    allow_failed_imports=False)

max_epochs = 40
train_cfg = dict(max_epochs=max_epochs)

model = dict(
    bbox_head=dict(num_classes=3, type="MySSDHead"),
    test_cfg=dict(max_per_img=300)
    )

data_root = './datasets/pklot/new-split/'
metainfo = {
    'classes': ('spaces', 'space-empty', 'space-occupied'),
    'palette': [
        (220, 220, 220),
        (20, 220, 60),
        (220, 20, 60),
    ]
}
annotation_file = 'pklot-annotations.json'
train_dataloader = dict(
    batch_size=8,
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

# train_dataloader = dict(
#     batch_size=1,
#     dataset=dict(
#         _delete_=True,
#         type='RepeatDataset',
#         times=1,
#         dataset=dict(
#             type={{_base_.dataset_type}},
#             metainfo=metainfo,
#             data_root=data_root,
#             ann_file='train/' + annotation_file,
#             data_prefix=dict(img='train/'),
#             pipeline=_base_.train_pipeline
#         )
#     )
# )
# val_dataloader = dict(
#     batch_size=1,
#     dataset=dict(
#         _delete_=True,
#         type='RepeatDataset',
#         times=1,
#         dataset=dict(
#             type={{_base_.dataset_type}},
#             metainfo=metainfo,
#             data_root=data_root,
#             ann_file='valid/' + annotation_file,
#             data_prefix=dict(img='valid/'),
#             pipeline=_base_.test_pipeline
#         )
#     )
# )
# test_dataloader = dict(
#     batch_size=1,
#     dataset=dict(
#         _delete_=True,
#         type='RepeatDataset',
#         times=1,
#         dataset=dict(
#             type={{_base_.dataset_type}},
#             metainfo=metainfo,
#             data_root=data_root,
#             ann_file='test/' + annotation_file,
#             data_prefix=dict(img='test/'),
#             pipeline=_base_.test_pipeline
#         )
#     )
# )

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'valid/' + annotation_file, metric=['bbox'])
test_evaluator = dict(ann_file=data_root + 'test/' + annotation_file, metric=['bbox'])
