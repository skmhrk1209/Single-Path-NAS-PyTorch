from torch import utils
from PIL import Image
import os


class ImageNet(utils.data.Dataset):

    def __init__(self, root, meta, transform=None):

        self.root = root
        self.transform = transform

        self.metas = []
        with open(meta) as file:
            for line in file.readlines():
                path, label = line.rstrip().split()
                self.metas.append((path, int(label)))

    def __len__(self):

        return len(self.metas)

    def __getitem__(self, index):

        path, label = self.metas[index]

        with open(os.path.join(self.root, path), 'rb') as file:
            image = Image.open(file).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label
