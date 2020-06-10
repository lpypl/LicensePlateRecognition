import torch.nn as nn
import torch as t
from model.PlateNumberDataSet import PlateNumberDataSet
from torch.utils.data import DataLoader
from torchvision import models
from torch import optim

train_path = '../digits_train/'
test_path = '../digits_test/'
trainset = PlateNumberDataSet(train_path)
testset = PlateNumberDataSet(test_path)
trainloader = DataLoader(trainset,
                         batch_size=3,
                         shuffle=True,
                         drop_last=False)
testloader = DataLoader(testset,
                        batch_size=1,
                        shuffle=False,
                        drop_last=False)


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

            optimizer.zero_grad()
            outputs = resnet34(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            # 每10步打印一下loss
            if i % 10 == 9:
                print('epoch: %d, %d loss: %f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    t.save(resnet34, './model.pkl')


def test():
    model = t.load('./model.pkl')
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


def main():
    train(epochs=30, num_class=67)
    test()


if __name__ == '__main__':
    main()
