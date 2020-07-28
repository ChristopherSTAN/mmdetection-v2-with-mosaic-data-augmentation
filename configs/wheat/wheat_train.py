_base_ = [
    '../_base_/datasets/mosaic_detection.py',  '../_base_/default_runtime.py'
]
model = dict(
    type='GFL',
    pretrained='https://s3.us-west-1.wasabisys.com/resnest/torch/resnest101-22405ba7.pth',
    backbone=dict(
        type='ResNeSt2',
        block='Bottleneck', layers=[3, 4, 23, 3],
        radix=2, groups=1, bottleneck_width=64,
        deep_stem=True, stem_width=64, avg_down=True,
        avd=True, avd_first=False, frozen_stages=1),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFLHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0))
)

train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=150)
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001, paramwise_cfg=dict(bias_decay_mult=0.))
# optimizer_config = dict(grad_clip=None)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
# lr_config = dict(
#     policy='CosineAnealing',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     min_lr=0.0001)
# total_epochs = 20
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12