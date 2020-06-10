import os
import numpy as np
from PIL import Image
import torch as t
from torch.utils import data
from torchvision import transforms as T
from matplotlib import pyplot as plt

plate_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵',
                '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏', '晋',
                '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']

transform = T.Compose([
    # T.Resize(128),  # 缩放图片Image
    # T.CenterCrop(128),  # 从图片中间切出图片
    T.ToTensor(),  # 将图片转成Tensor，归一化为 [0,1]
])


class PlateNumberDataSet(data.Dataset):
    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.labels = [plate_labels.index(img_name.split('_')[0]) for img_name in imgs]
        self.transforms = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_label = self.labels[index]
        img_data = Image.open(img_path)
        if self.transforms:
            img_data = self.transforms(img_data)
        return img_data, img_label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    img_root = '../digits/'
    dataset = PlateNumberDataSet(img_root)
    to_img = T.ToPILImage()
    img, label = dataset[0]
    img = to_img(img)
    img.show()
    print(len(plate_labels))
    for img, label in dataset:
        print(img.size(), label)
