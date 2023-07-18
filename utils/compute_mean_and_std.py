import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from tqdm import tqdm


def compute_ffhq_statistics(image_size: int = 32):

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    dataset = ImageFolder(
        root='E:/GitHub/igd-slbt-master-thesis/data/FFHQ/images_dummy',
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs, _ in tqdm(dataloader):

        psum += inputs.sum(axis=[0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])

    # pixel count
    count = len(dataset) * image_size * image_size

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    return total_mean, total_std


if __name__ == '__main__':

    mean, std = compute_ffhq_statistics(image_size=32)
    print("MEAN", mean)
    print("STD", std)