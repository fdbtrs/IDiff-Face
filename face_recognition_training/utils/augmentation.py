import torchvision.transforms as transforms
import torch
import logging
import moco.loader
from utils.rand_augment import RandAugment
from utils.FAA_policy import IResNet50CasiaPolicy, ReducedImageNetPolicy

normalize_moco = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


# MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
aug_plus = [
    transforms.RandomResizedCrop(112, scale=(0.2, 1.0)),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_moco,
]

# MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
aug_default = [
    transforms.RandomResizedCrop(112, scale=(0.2, 1.0)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_moco,
]

to_tensor = [transforms.ToTensor(), normalize]

only_normalize = [normalize]

online_RA_2_16 = [
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=4, magnitude=16),
    normalize,
]

aug_h_flip = [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]

aug_rand_4_16 = [
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=4, magnitude=16),
    transforms.ToTensor(),
    normalize,
]

aug_online_rand_4_16 = [
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=4, magnitude=16),
]

aug_rand_2_9 = [
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    normalize,
]

aug_rand_4_24 = [
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=4, magnitude=24),
    transforms.ToTensor(),
    normalize,
]

aug_CASIA_FAA = [
    transforms.RandomHorizontalFlip(),
    IResNet50CasiaPolicy(),
    transforms.ToTensor(),
    normalize,
]

aug_ImgNet_FAA = [
    transforms.RandomHorizontalFlip(),
    ReducedImageNetPolicy(),
    transforms.ToTensor(),
    normalize,
]


def get_randaug(n, m):
    """return RandAugment transforms with
    n: number of operations
    m: magnitude
    """
    return [
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=n, magnitude=m),
        transforms.ToTensor(),
        normalize,
    ]


def select_x_operation(x):
    """enable only the x operation to RandAug
    x: string of the available augmentation
    """
    return [
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=1, magnitude=9, available_aug=x),
        transforms.ToTensor(),
        normalize,
    ]


def get_conventional_aug_policy(aug_type, operation="none", num_ops=0, mag=0):
    """get geometric and color augmentations
    args:
        aug_type: string defining augmentation type
        operation: RA augmentation operation under testing
        num_ops: number of sequential operations under testing
        mag: magnitude under testing
    return:
        augmentation policy
    """
    aug = aug_type.lower()
    if aug == "gan_hf" or aug == "nogan_hf" or aug == "hf":
        augmentation = aug_h_flip
    elif aug == "gan_ra_4_16" or aug == "nogan_ra_4_16" or aug == "ra_4_16":
        augmentation = aug_rand_4_16
    elif aug == "moco":
        augmentation = aug_default
    elif aug == "faa_casia":
        augmentation = aug_CASIA_FAA
    elif aug == "faa_imgnet":
        augmentation = aug_ImgNet_FAA
    elif aug == "num_mag_exp":
        augmentation = get_randaug(num_ops, mag)
    elif aug == "aug_operation_exp":
        logging.info("Augmentation under testing: " + operation)
        augmentation = select_x_operation(operation)
    elif aug == "totensor":
        augmentation = to_tensor
    elif aug == "mag_totensor":
        augmentation = [transforms.ToTensor()]
    else:
        logging.error("Unknown augmentation method: {}".format(aug_type))
        exit()
    return transforms.Compose(augmentation)


""" default simCLR augmentation """
s = 1
color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
aug_simCLR = [
    transforms.RandomResizedCrop(
        size=[112, 112], scale=(0.75, 1)
    ),  # scale not given in default SimCLR augmentation
    transforms.RandomHorizontalFlip(),  # with 0.5 probability
    transforms.RandomApply([color_jitter], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    normalize,  # not given in default SimCLR augmentation
]


class GaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )
