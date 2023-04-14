import os
from PIL import Image
from torch.utils.data import Dataset

class ImageFolder(Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

        self.image_folder = os.path.join(root, 'image')
        self.mask_folder = os.path.join(root, 'mask')
        self.img_list = [os.path.splitext(f)[0] for f in os.listdir(self.image_folder) if f.endswith('.jpg')]

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_path = os.path.join(self.image_folder, img_name + '.jpg')
        gt_path = os.path.join(self.mask_folder, img_name + '.png')

        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')

        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img_list)