# The new config inherits a base config to highlight the necessary modification
#_base_ = 'configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'
#_base_ = 'configs/mask_rcnn/mask-rcnn_r50_fpn_ms-poly-3x_coco.py'
_base_ = '../../configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'

max_epochs = 20
train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=1)

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=3), mask_head=dict(num_classes=3)
    ),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=1000
        ),
        rpn=dict(
            max_per_img=1000
        ),
    )
)

# Modify dataset related settings
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
val_evaluator = dict(ann_file=data_root + 'valid/' + annotation_file, metric=['bbox'], proposal_nums=[1000, 1000, 1000], use_mp_eval=True)
test_evaluator = dict(ann_file=data_root + 'test/' + annotation_file, metric=['bbox'], proposal_nums=[1000, 1000, 1000], use_mp_eval=True)

# We can use the pre-trained Mask RCNN model to obtain higher performance
#load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
#load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth'
