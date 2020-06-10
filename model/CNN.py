import torch.nn as nn
import torch as t
from model.PlateNumberDataSet import PlateNumberDataSet
from torch.utils.data import DataLoader
from torchvision import models
from torch import optim

digit_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
han_labels = ['粤', '云', '浙', '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪', '吉', '冀', '津', '晋', '京', '辽', '鲁',
              '蒙', '闽', '宁', '青', '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫']

# train_path = '../dataset/数字和字母训练集/'
# test_path = '../dataset/数字和字母测试集/'
train_path = '../dataset/汉字训练集/'
test_path = '../dataset/汉字测试集/'
trainset = PlateNumberDataSet(train_path, han=True)
testset = PlateNumberDataSet(test_path, han=True)
trainloader = DataLoader(trainset,
                         batch_size=32,
                         shuffle=True,
                         drop_last=False)
testloader = DataLoader(testset,
                        batch_size=1,
                        shuffle=False,
                        drop_last=False)

"""
digits训练30个epoch，batch_size=128正确率97.6%，loss仍有下降空间。
汉字训练60个epoch batch_size=32，正确率95.3%，loss仍有下降空间。
"""


def train(epochs, num_class):
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

            # 每100步打印一下loss
            if i % 32 == 31:
                print('epoch: %d, batch: %d loss: %f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    t.save(resnet34, './han_model.pkl')


def digit_test():
    model = t.load('./digit_model.pkl')
    total = 0
    correct = 0
    for data in testloader:
        image, label = data
        if t.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        output = model(image)
        print("predict：%d,true：%d" % (t.argmax(output), label))
        if t.argmax(output) == label:
            correct = correct + 1
        total = total + 1
    print("正确率为：", correct / total)


def han_test():
    model = t.load('./han_model.pkl')
    total = 0
    correct = 0
    for data in testloader:
        image, label = data
        if t.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        output = model(image)
        print("predict：%d,true：%d" % (t.argmax(output), label))
        if t.argmax(output) == label:
            correct = correct + 1
        total = total + 1
    print("正确率为：", correct / total)


def main():
    train(epochs=60, num_class=len(han_labels))
    # digit_test()
    han_test()


if __name__ == '__main__':
    main()
