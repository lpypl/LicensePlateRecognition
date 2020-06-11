import torch.nn as nn
import torch as t
from CNNCharReco.PlateNumberDataSet import PlateNumberDataSet
from torch.utils.data import DataLoader
from torchvision import models
from torch import optim
from torchvision import transforms as T
from PIL import Image
import numpy as np

digit_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
han_labels = ['粤', '云', '浙', '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁',
              '蒙', '闽', '宁', '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫']

digit_train_path = '../dataset/数字和字母训练集/'
digit_test_path = '../dataset/数字和字母测试集/'
han_train_path = '../dataset/汉字训练集/'
han_test_path = '../dataset/汉字测试集/'

digit_trainset = PlateNumberDataSet(digit_train_path, han=False)
digit_testset = PlateNumberDataSet(digit_test_path, han=False)
han_trainset = PlateNumberDataSet(han_train_path, han=True)
han_testset = PlateNumberDataSet(han_test_path, han=True)

digit_trainloader = DataLoader(digit_trainset,
                               batch_size=128,
                               shuffle=True,
                               drop_last=False)
digit_testloader = DataLoader(digit_testset,
                              batch_size=1,
                              shuffle=False,
                              drop_last=False)

han_trainloader = DataLoader(han_trainset,
                             batch_size=32,
                             shuffle=True,
                             drop_last=False)
han_testloader = DataLoader(han_testset,
                            batch_size=1,
                            shuffle=False,
                            drop_last=False)

"""
digits训练30个epoch，batch_size=128正确率97.6%，loss仍有下降空间。
汉字训练60个epoch batch_size=32，正确率95.3%，loss仍有下降空间。
"""


def train(epochs, num_class, trainloader, model_save_path):
    resnet34 = models.squeezenet1_1(pretrained=True, num_classes=1000)
    resnet34.fc = nn.Linear(512, num_class)

    if t.cuda.is_available():
        resnet34.cuda()

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(resnet34.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if t.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = resnet34(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            # 每32个batch打印一下loss
            if i % 32 == 31:
                print('epoch: %d, batch: %d loss: %f' % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0.0

    t.save(resnet34, model_save_path)


def model_test(testloader, model_path):
    model = t.load(model_path, map_location='cpu')
    total = 0
    correct = 0
    for data in testloader:
        image, label = data
        output = model(image)
        print("predict：%d,true：%d" % (t.argmax(output), label))
        if t.argmax(output) == label:
            correct = correct + 1
        total = total + 1
    print("正确率为：", correct / total)


def predict(pil_img, model_path, kind='digit'):
    model = t.load(model_path, map_location='cpu')
    transform = T.Compose([
        T.Resize((28, 28)),  # 缩放图片
        T.ToTensor(),  # 转为tensor
    ])
    # 把图像处理为标准格式
    img = transform(pil_img)
    img = img.unsqueeze(0)
    output = model(img)
    index = t.argmax(output)
    if kind == 'digit':
        result = digit_labels[index]
    elif kind == 'han':
        result = han_labels[index]
    return result


def main():
    # 训练模型
    # train(epochs=30, num_class=len(digit_labels), trainloader=digit_trainloader, model_save_path='./digit_model.pkl')
    # train(epochs=60, num_class=len(han_labels), trainloader=han_trainloader, model_save_path='./han_model.pkl')

    # 字母和数字测试
    # model_test(testloader=digit_testloader, model_path='./digit_model.pkl')
    # 汉字测试
    # model_test(testloader=han_testloader, model_path='./han_model.pkl')

    # 模型预测
    digit_img = Image.open('../dataset/单字符车牌/10/000001.jpg')
    reuslt = predict(pil_img=digit_img, model_path='./digit_model.pkl', kind='digit')
    print(reuslt)
    han_img = Image.open('../dataset/单字符车牌/50/000001.jpg')
    reuslt = predict(pil_img=han_img, model_path='./han_model.pkl', kind='han')
    print(reuslt)


if __name__ == '__main__':
    main()
