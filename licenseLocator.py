from licenseLocatorHyperLPR import LicenseLocator as model1
from Mask_RCNN.carPlateLocation import LicenseLocator as model2
import cv2


class LicenseLocator:
    def __init__(self, bgrImage):
        self.bgrImage = bgrImage
        self.fitComplete = False
        self.model = None

    def fit(self):
        if self.fitComplete:
            return True

        self.model = model1(self.bgrImage)
        if not self.model.fit():
            self.model = model2(self.bgrImage)
        self.fitComplete = True

    def getRectImage(self):
        """
        获取带车牌轮廓的图片
        :return: 若寻找失败，返回None
        """
        self.fit()
        return self.model.getRectImage()

    def getLicenseImage(self):
        """
        获取车牌图片
        :return: 若寻找失败，返回None
        """
        self.fit()
        return self.model.getLicenseImage()


if __name__ == '__main__':
    bgrImage = cv2.imread('dataset/example28.bmp')
    ll = LicenseLocator(bgrImage)
    rectImage = ll.getRectImage()
    if rectImage is not None:
        cv2.imshow('', rectImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    licenseImage = ll.getLicenseImage()
    if licenseImage is not None:
        cv2.imshow('', licenseImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        from charSegment import CharSplitter
        import numpy as np

        splitter = CharSplitter(licenseImage)
        grayImage = splitter.getGrayImage()
        cv2.imshow('W1', grayImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        binaryImage = splitter.getBinaryImage()
        cv2.imshow('W1', binaryImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        segmentImage = splitter.getSegmentImage()
        cv2.imshow('W1', segmentImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        charImages = splitter.getCharImages()
        imgs = np.hstack(charImages)
        cv2.imshow('W1', imgs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

