import HyperLPR.HyperLPRLite as pr
import cv2
from ImageCorrection import imageCorrection


class LicenseLocator:
    def __init__(self, bgrImage):
        self.bgrImage = bgrImage.copy()
        self.rect = None
        self.fitComplete = False

    def fit(self):
        if self.fitComplete:
            return True
        model = pr.LPR("HyperLPR/model/cascade.xml", "HyperLPR/model/model12.h5", "HyperLPR/model/ocr_plate_all_gru.h5")
        for pstr, confidence, rect in model.SimpleRecognizePlateByE2E(self.bgrImage):
            if confidence > 0.7:
                self.rect = (int(rect[0]), int(rect[1]), int(rect[0] + rect[2]), int(rect[1] + rect[3]))
                # print("plate_str:", pstr)
                # print("plate_confidence", confidence)
        if self.rect is None:
            return False
        else:
            self.fitComplete = True
            return True

    def getRectImage(self):
        """
        获取带车牌轮廓的图片
        :return: 若寻找失败，返回None
        """
        if not self.fit():
            return None
        image = self.bgrImage.copy()
        cv2.rectangle(image, (self.rect[0], self.rect[1]), (self.rect[2], self.rect[3]), (0, 0, 255), 2,
                      cv2.LINE_AA)
        return image

    def getLicenseImage(self):
        """
        获取车牌图片
        :return: 若寻找失败，返回None
        """
        if not self.fit():
            return None
        licenseImage = self.bgrImage[self.rect[1]:self.rect[3], self.rect[0]:self.rect[2]]
        return imageCorrection(licenseImage)


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

