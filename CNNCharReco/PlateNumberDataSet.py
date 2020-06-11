import os
import numpy as np
from PIL import Image
import torch as t
from torch.utils import data
from torchvision import transforms as T
from matplotlib import pyplot as plt

digit_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
han_labels = ['粤', '云', '浙', '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁',
              '蒙', '闽', '宁', '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫']

transform = T.Compose([
    T.Resize((28, 28)),  # 缩放图片Image
    T.ToTensor(),  # 将图片转成Tensor，归一化为 [0,1]
])


class PlateNumberDataSet(data.Dataset):
    def __init__(self, root, transforms=None, han=False):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        if han:
            self.labels = [han_labels.index(img_name.split('_')[0]) for img_name in imgs]
        else:
            self.labels = [digit_labels.index(img_name.split('_')[0]) for img_name in imgs]
        if transforms:
            self.transforms = transforms
        else:
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
    img_root = '../dataset/数字和字母测试集/'
    dataset = PlateNumberDataSet(img_root, han=False)
    to_img = T.ToPILImage()
    img, label = dataset[0]
    img = to_img(img)
    img.show()
    print(len(digit_labels))
    for img, label in dataset:
        print(img.size(), label)
