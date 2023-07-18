import math

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from diffusion.ddpm import compute_beta_schedule, precompute_schedule_constants

T = 1000
d = 10

x = torch.linspace(0, 1, T // d)

paths = [
    "E:/GitHub/igd-slbt-master-thesis/additional/DALL-2/raw/DALL·E 2022-09-29 10.20.48.png",
    "E:/GitHub/igd-slbt-master-thesis/data/FFHQ/images/67863.png",
    "E:/GitHub/igd-slbt-master-thesis/data/FFHQ/images/66912.png"
]

image_path = paths[0]


def clip_and_cast(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def compute_alphabars(schedule_type, **kwargs):
    betas = compute_beta_schedule(T=T, schedule_type=schedule_type, **kwargs)
    alphabars = torch.cumsum(torch.log(1 - betas), dim=0).exp()
    return alphabars


def create_image_grid_for_schedule(path, schedule_type, **kwargs):
    image = Image.open(path)

    image = image.resize((128, 128))
    image = np.array(image)
    image = torch.from_numpy(image) / 255.0

    image = (image * 2) - 1

    betas = compute_beta_schedule(T=T, schedule_type=schedule_type, **kwargs)
    constants = precompute_schedule_constants(betas)

    noisy_images = [image * 127.5 + 127.5]
    for t in list(range(0, T, T // d)) + [T-1]:
        eps = torch.randn(image.shape)

        mean = constants['sqrt_alpha_bars'][t] * image
        sd = constants['sqrt_one_minus_alpha_bars'][t]

        x_noisy = mean + sd * eps
        x_noisy = (x_noisy * 127.5) + 127.5
        noisy_images.append(x_noisy)

    return torch.cat(noisy_images, dim=1)


# ======================== Plot that varies parameter k ================================================================
f, axes = plt.subplots(2, 3)

ax = axes[0, 0]

# LINEAR
alphabars_linear = compute_alphabars('linear')[::d]
ax.plot(x, alphabars_linear, label="linear", alpha=0.5, linestyle='dashed')
alphabars_warped = compute_alphabars('cosine')[::d]
ax.plot(x, alphabars_warped, label=f"C(1, 0.0)", alpha=0.5, linestyle='dashed')

schedule_images = [create_image_grid_for_schedule(image_path, 'linear')]

# WARPED COSINE for k's
for k in [0.5, 1.5, 2, 4, 8]:
    alphabars_warped = compute_alphabars('cosine_warped', k=k)[::d]
    ax.plot(x, alphabars_warped, label=f"C({k}, 0.0)")

    schedule_images.append(create_image_grid_for_schedule(image_path, 'cosine_warped', k=k))

ax.legend()

all_images_0 = torch.cat(schedule_images, dim=0)
all_images_0 = all_images_0.numpy()

# ======================== Plot that varies parameter beta_min =========================================================

ax = axes[0, 1]
k = 1

schedule_images = []

ax.plot(x, alphabars_linear, label="linear", alpha=0.5, linestyle='dashed')
alphabars_warped = compute_alphabars('cosine')[::d]
ax.plot(x, alphabars_warped, label=f"C(1, 0.0)", alpha=0.5, linestyle='dashed')

for beta_min in [1, 2, 3, 4, 5, 6]:
    beta_min *= 1e-3
    alphabars_warped = compute_alphabars('cosine_warped', k=k, beta_min=beta_min)[::d]
    ax.plot(x, alphabars_warped, label=f"C({k}, {beta_min})")

    schedule_images.append(create_image_grid_for_schedule(image_path, 'cosine_warped', k=k, beta_min=beta_min))

ax.legend()

all_images_1 = torch.cat(schedule_images, dim=0)
all_images_1 = all_images_1.numpy()

# ======================== Plot that varies both =======================================================================

ax = axes[0, 2]
k = 2

schedule_images = []

ax.plot(x, alphabars_linear, label="linear", alpha=0.5, linestyle='dashed')
alphabars_warped = compute_alphabars('cosine')[::d]
ax.plot(x, alphabars_warped, label=f"C(1, 0.0)", alpha=0.5, linestyle='dashed')

for beta_min in [0, 1, 2, 4, 8, 16]:
    k = 3
    beta_min *= 1e-4
    alphabars_warped = compute_alphabars('cosine_warped', k=k, beta_min=beta_min)[::d]
    ax.plot(x, alphabars_warped, label=f"C({k}, {beta_min})")

    schedule_images.append(create_image_grid_for_schedule(image_path, 'cosine_warped', k=k, beta_min=beta_min))

ax.legend()

all_images_2 = torch.cat(schedule_images, dim=0)
all_images_2 = all_images_2.numpy()

ax.legend()

axes[0, 0].set_title(r"Varying the warping factor $k$")
axes[0, 1].set_title(r"Varying the minimal beta value $\beta_{min}$")
axes[0, 2].set_title("Combination of both")

axes[0, 0].set_xlabel("timestep $t$")
axes[0, 1].set_xlabel("timestep $t$")
axes[0, 2].set_xlabel("timestep $t$")

axes[0, 0].set_ylabel(r"$\bar{\alpha}_t$")

all_images0 = clip_and_cast(all_images_0)
all_images1 = clip_and_cast(all_images_1)
all_images2 = clip_and_cast(all_images_2)

axes[1, 0].imshow(all_images0)
axes[1, 1].imshow(all_images1)
axes[1, 2].imshow(all_images2)

axes[1, 0].axis('off')
axes[1, 1].axis('off')
axes[1, 2].axis('off')

im = Image.fromarray(all_images_0.astype(np.uint8))
im.save(f"all_images_0.png")

im = Image.fromarray(all_images_1.astype(np.uint8))
im.save(f"all_images_1.png")

im = Image.fromarray(all_images_2.astype(np.uint8))
im.save(f"all_images_2.png")

for ax in axes[0]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.show()
