import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil

digit_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
han_labels = ['粤', '云', '浙', '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁',
              '蒙', '闽', '宁', '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫']


# 数字和字母 0~33
# 汉字 34~64
def generate_train_and_test_set():
    for i in range(0, 34):
        imgs = os.listdir('../dataset/单字符车牌/' + str(i) + '/')
        img_label = digit_labels[i]
        for index, img_name in enumerate(imgs, 1):
            src = '../dataset/单字符车牌/' + str(i) + '/' + img_name
            new_name = img_label + '_' + str(index) + '.jpg'
            dst = '../dataset/单字符车牌/数字和字母/' + new_name
            shutil.copy(src, dst)
    for i in range(34, 65):
        imgs = os.listdir('../dataset/单字符车牌/' + str(i) + '/')
        img_label = han_labels[i - 34]
        for index, img_name in enumerate(imgs, 1):
            src = '../dataset/单字符车牌/' + str(i) + '/' + img_name
            new_name = img_label + '_' + str(index) + '.jpg'
            dst = '../dataset/单字符车牌/汉字/' + new_name
            shutil.copy(src, dst)


def main():
    generate_train_and_test_set()


if __name__ == '__main__':
    main()
