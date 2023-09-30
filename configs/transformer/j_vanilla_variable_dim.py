import wandb
wandb.init(project='ViViT2')

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='LST',
        hidden_dim=64,
        dim_mul_layers=(4, 7),
        temporal_pooling=True,
        sliding_window=False,
        # graph_cfg=dict(layout='nturgb+d', mode='spatial'),
    ),
    cls_head=dict(type='TRHead', num_classes=60, in_channels=256, dropout=0.))

dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_3danno.pkl'

clip_len = 64
mode = 'zero'
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='RandomRot', theta=0.2),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=clip_len),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2, mode=mode),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=clip_len, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2, mode=mode),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=clip_len, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2, mode=mode),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=8,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xsub_train'),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xsub_val'))

# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=3.0, norm_type=2))
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
# optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=10,
    warmup_ratio=1.0 / 100,
    min_lr_ratio=1e-6)
total_epochs = 80
checkpoint_config = dict(interval=1)
evaluation = dict(interval=2, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook'), dict(type='WandbLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/lst/ntu60_xsub_3dkp/j_vanilla_variable_dim/9.30-0'
find_unused_parameters = False
auto_resume = False
seed = 88

