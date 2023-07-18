from PIL import Image
import os

src_path = "E:/GitHub/igd-slbt-master-thesis/data/FFHQ/images_dummy/images"
tar_path = "E:/GitHub/igd-slbt-master-thesis/data/FFHQ/images_32px/images"
os.makedirs(tar_path, exist_ok=True)


def resize():
    for file in os.listdir(src_path):

        img = Image.open(os.path.join(src_path, file))
        img = img.resize((32, 32), Image.ANTIALIAS)

        img.save(os.path.join(tar_path, file), 'PNG', quality=90)


if __name__ == '__main__':
    resize()