import cv2
import numpy as np
from PIL import Image
import torch as t
from torchvision import transforms as T


class LicenseReco:
    def __init__(self, digit_model_path, han_model_path):
        self.digit_model = t.load(digit_model_path, map_location='cpu')
        self.han_model = t.load(han_model_path, map_location='cpu')
        self.digit_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                             'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.han_labels = ['粤', '云', '浙', '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋',
                           '京', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫']
        self.transforms = T.Compose([
            T.Pad(padding=(13, 0, 13, 0), fill=0),
            T.Resize((35, 35)),
            T.CenterCrop((28, 28)),
            T.Resize((28, 28)),  # 缩放图片Image
            T.ToTensor(),  # 将图片转成Tensor，归一化为 [0,1]
        ])

    def predict(self, charImages: list) -> str:
        to_img = T.ToPILImage()
        result = []
        han_img = charImages[0]
        han_img = Image.fromarray(han_img)
        han_img = han_img.convert(mode='RGB')
        # han_img.show()
        han_img = self.transforms(han_img)
        # img = to_img(han_img)
        # img.show()
        han_img = han_img.unsqueeze(0)
        output = self.han_model(han_img)
        index = t.argmax(output)
        result.append(self.han_labels[index])

        for digit_img in charImages[1:]:
            digit_img = Image.fromarray(digit_img)
            digit_img = digit_img.convert(mode='RGB')
            # digit_img.show()
            digit_img = self.transforms(digit_img)
            # img = to_img(digit_img)
            # img.show()
            digit_img = digit_img.unsqueeze(0)
            output = self.digit_model(digit_img)
            index = t.argmax(output)
            result.append(self.digit_labels[index])

        return ''.join(result)
