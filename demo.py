from pretreatment import *
from licenseLocator import *
from charSegment import *
import cv2
from licenseReco import LicenseReco

if __name__ == '__main__':
    bgrImage = cv2.imread('dataset/example28.bmp')

    # # Pretreatment
    # pt = Pretreatment(bgrImage)
    # cv2.imshow('', pt.getGrayImage())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # cv2.imshow('', pt.getHistEqImage())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # cv2.imshow('', pt.getMedianImage())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # LicenseLocator
    ll = LicenseLocator(bgrImage)
    rectImage = ll.getRectImage()
    if rectImage is not None:
        cv2.imshow('', rectImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    licenseImage = ll.getLicenseImage()
    if licenseImage is not None:
        # cv2.imshow('', licenseImage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # CharSplitter
        splitter = CharSplitter(licenseImage)
        grayImage = splitter.getGrayImage()
        # cv2.imshow('W1', grayImage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        binaryImage = splitter.getBinaryImage()
        cv2.imshow('W1', binaryImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        segmentImage = splitter.getSegmentImage()
        cv2.imshow('W1', segmentImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        charImages = splitter.getCharImages()

        # 测试
        reco = LicenseReco(digit_model_path='./CNNCharReco/digit_model.pkl',
                           han_model_path='./CNNCharReco/han_model.pkl')
        result = reco.predict(charImages)
        print(result)
        imgs = np.hstack(charImages)
        cv2.imshow('', imgs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()