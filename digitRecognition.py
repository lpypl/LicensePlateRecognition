import cv2
import imageProcess as ip
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D
from keras.utils import to_categorical
import numpy as np


def load_trainset():
    import os
    dirname = r'/home/lpy/PythonProjects/LicensePlateRecognition/digits'
    files = os.listdir(dirname)
    digits = []
    labels = []
    for fname in files:
        fpath = f'{dirname}/{fname}'
        labels.append(fname[0])
        digit = cv2.imread(fpath)
        digit = ip.img2gray(digit)
        grayimg = digit.copy()
        # histogram equalization
        grayimg = cv2.equalizeHist(grayimg)
        # smooth
        for _ in range(3):
            grayimg = cv2.medianBlur(grayimg, 3)
            grayimg = cv2.blur(grayimg, (3, 3))
        grayimg = ip.gray2binary(grayimg, 170)
        digits.append(grayimg)
    return digits, labels


def cnn_train(X_train, y_train, X_test, y_test):
    input_shape = X_train.shape[1:]
    out_units = len(y_train[0])
    model = Sequential()
    model.add(Conv2D(25, (5, 5), input_shape=input_shape))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(25, (5, 5)))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(25, (5, 5)))
    model.add(MaxPool2D(2, 2))
    model.add(Flatten())
    model.add(Dense(units=500))
    model.add(Activation('relu'))
    model.add(Dense(units=500))
    model.add(Activation('relu'))
    model.add(Dense(units=out_units))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10)

    score = model.evaluate(X_test, y_test)
    print('Total loss on Testing Set:', score[0])
    print('Accuracy of Testing Set:', score[1])


def show_digits():
    digits, labels = load_trainset()
    for digit in digits:
        cv2.imshow('W', digit)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def cnn_test():
    digits, labels = load_trainset()
    row, col = digits[0].shape
    # resize - inplace
    # reshape - non inplace
    for img in digits:
        img.resize((1, row, col))
    X_train = np.array(digits)
    y_train = np.array([ord(c) for c in labels])
    y_train = to_categorical(y_train)
    elecnt = X_train.shape[0]
    train_sample = np.random.choice(elecnt, elecnt//2, replace=False)
    real_X_train = X_train[train_sample]
    real_y_train = y_train[train_sample]

    test_sample = np.random.choice(elecnt, elecnt//2, replace=False)
    X_test = X_train[test_sample]
    y_test = y_train[test_sample]
    cnn_train(real_X_train, real_y_train, X_test, y_test)


def main():
    cnn_test()


if __name__ == '__main__':
    main()
