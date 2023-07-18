import os

architecture = "resnet50"

dataset_folder = "specify the path to the aligned images folder"

MODELS = [
    "unet-cond-ca-bs512-150K",
    "unet-cond-ca-bs512-150K-cpd25",
    "unet-cond-ca-bs512-150K-cpd50"
]

EMBEDDING_TYPE = [
    "random_synthetic_uniform_5000",
    "random_synthetic_learned_5000",
    "random_synthetic_extracted_5000"
]

model = MODELS[0]
embedding_type = EMBEDDING_TYPE[0]

width = 0
depth = 0

batch_size = 128  # 256
workers = 8  # 32
embedding_size = 512
learning_rate = 0.1
momentum = 0.9
weight_decay = 5e-4

global_step = 0  # to resume
start_epoch = 0

s = 64.0
m = 0.35
loss = "CosFace"
dropout_ratio = 0.4

augmentation = "ra_4_16"  # hf, ra_4_16

print_freq = 50
val_path = "/data/Biometrics/database/faces_emore"  # "/data/fboutros/faces_emore"
val_targets = ["lfw", "agedb_30", "cfp_fp", "calfw", "cplfw"]

auto_schedule = True
num_epoch = 200
schedule = [22, 30, 35]


def lr_step_func(epoch):
    return (
        ((epoch + 1) / (4 + 1)) ** 2
        if epoch < -1
        else 0.1 ** len([m for m in schedule if m - 1 <= epoch])
    )


lr_func = lr_step_func
