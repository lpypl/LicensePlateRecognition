from sklearn.model_selection import train_test_split
import os
import shutil

plate_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵',
                '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏', '晋',
                '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']


def generate_train_and_test_set(img_root_path, to_train_path, to_test_path):
    imgs = os.listdir(img_root_path)
    labels = [plate_labels.index(img_name.split('_')[0]) for img_name in imgs]

    X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2, shuffle=True, random_state=1)
    for img in X_train:
        in_path = os.path.join(img_root_path, img)
        to_path = os.path.join(to_train_path, img)
        shutil.copy(in_path, to_path)
    for img in X_test:
        in_path = os.path.join(img_root_path, img)
        to_path = os.path.join(to_test_path, img)
        shutil.copy(in_path, to_path)


if __name__ == '__main__':
    img_root_path = '../digits/'
    to_train_path = '../digits_train/'
    to_test_path = '../digits_test/'
    generate_train_and_test_set(img_root_path, to_train_path, to_test_path)
