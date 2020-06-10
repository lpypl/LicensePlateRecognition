import cv2


class Pretreatment:
    def __init__(self, bgrImage):
        self.bgrImage = bgrImage

    def getGrayImage(self):
        return cv2.cvtColor(self.bgrImage, cv2.COLOR_BGR2GRAY)

    def getHistEqImage(self):
        return cv2.equalizeHist(self.getGrayImage())

    def getMedianImage(self):
        return cv2.medianBlur(self.getHistEqImage(), 3)


if __name__ == '__main__':
    image = cv2.imread('dataset/example20.bmp')
    pt = Pretreatment(image)
    cv2.imshow('', pt.getGrayImage())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('', pt.getHistEqImage())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('', pt.getMedianImage())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
