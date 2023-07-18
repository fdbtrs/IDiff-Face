# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random


class TwoCropsTransform:
    """Augment query with base transform and key with key augmentation.
    If key augmentation == None, key is augmented with base transform"""

    def __init__(self, base_transform, key_augmentation=None):
        self.base_transform = base_transform
        self.key_augment = (
            base_transform if key_augmentation is None else key_augmentation
        )

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.key_augment(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
