import wandb
wandb.init(project='ViViT2')

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='ViViT2',
        graph_cfg=dict(layout='nturgb+d', mode='spatial'),
        max_position_embeddings_1=26,  # 25*40+1=1001
        max_position_embeddings_2=101,
        # dropout=0.1,
        dim=256,
        # graph_cfg=dict(layout='nturgb+d', mode='random', num_filter=8, init_off=.04, init_std=.02),
    ),
    cls_head=dict(type='vit2Head', num_classes=60, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_3danno.pkl'
clip_len = 100
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='RandomScale', scale=0.1),
    dict(type='RandomRot'),
    # dict(type='RandomRot', theta=0.2),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=clip_len),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=clip_len, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=clip_len, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=24,
    workers_per_gpu=8,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xsub_train')),
    # train=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xsub_train'),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xsub_val'))

# optimizer
# optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005, amsgrad=False)
optimizer = dict(type='Adam', lr=0.001)
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0004, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 100
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/transformer/j2/9.6-tm2-3'

auto_resume = False
seed = 88
