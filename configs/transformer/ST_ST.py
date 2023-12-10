import wandb
wandb.init(project='ViViT2')

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='ST_ST',
    ),
    cls_head=dict(type='vit2Head', num_classes=60, in_channels=512, dropout=0.))

dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_3danno.pkl'

clip_len = 64
# mode = 'zero'
# train_pipeline = [
#     dict(type='PreNormalize3D'),
#     dict(type='RandomScale', scale=0.1),
#     dict(type='RandomRot', theta=0.3),
#     dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
#     dict(type='SampleFrames', clip_len=clip_len, frame_interval=sample_rate,
#          out_of_bound_opt='repeat_last', keep_tail_frames=True),
#     dict(type='PoseDecode'),
#     dict(type='FormatGCNInput', num_person=2, mode=mode),
#     dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['keypoint'])
# ]
# val_pipeline = [
#     dict(type='PreNormalize3D'),
#     dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
#     dict(type='SampleFrames', clip_len=clip_len, frame_interval=sample_rate,
#          out_of_bound_opt='repeat_last', keep_tail_frames=True),
#     dict(type='PoseDecode'),
#     dict(type='FormatGCNInput', num_person=2, mode=mode),
#     dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['keypoint'])
# ]
# test_pipeline = [
#     dict(type='PreNormalize3D'),
#     dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
#     dict(type='SampleFrames', clip_len=clip_len, frame_interval=sample_rate,
#          out_of_bound_opt='repeat_last', keep_tail_frames=True, num_clips=10),
#     dict(type='PoseDecode'),
#     dict(type='FormatGCNInput', num_person=2, mode=mode),
#     dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['keypoint'])
# ]
mode = 'zero'
train_pipeline = [
    dict(type='PreNormalize3D'),
    # dict(type='RandomRot', theta=0.2),
    dict(type='RandomScale', scale=0.1),
    dict(type='RandomRot'),
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
    videos_per_gpu=24,
    workers_per_gpu=8,
    test_dataloader=dict(videos_per_gpu=8),
    train=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xsub_train'),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xsub_val'))

# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=3.0, norm_type=2))
# learning policy
# lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=10,
    warmup_ratio=1.0 / 100,
    min_lr_ratio=1e-6)
total_epochs = 100
checkpoint_config = dict(interval=1)
evaluation = dict(interval=4, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook'), dict(type='WandbLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/lst/ntu60_xsub_3dkp/j_vanilla_variable_dim/12.8-ST_ST-basic2'
find_unused_parameters = True
auto_resume = False
seed = 88
