_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/je_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=30)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[64.34, 64.34, 64.34],
    std=[42.92, 42.92, 42.92],
    bgr_to_rgb=False,
    size=None,
    size_divisor=64,
    pad_val=0,
    seg_pad_val=0)


model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=256, stride=170))



vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends,save_dir='tensorboard_logs' ,name='visualizer')

optim_wrapper = dict(accumulative_counts=8)
