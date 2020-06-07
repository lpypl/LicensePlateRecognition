import cv2
import sys
from basicAlgorithm import *


class CharSplitter:
    def __init__(self, licenseBGRImage):
        # resize
        self.bgrImage = cv2.resize(licenseBGRImage, (500, 130), cv2.INTER_AREA)

    def __crop_vertical_border(self, binaryImg):
        """
        借助在Y轴上的投影裁减掉上方和下方的边界
        :param binaryImg: 二值图像
        :return: 裁减掉上方和下方的边界的二值图像
        """
        binaryImg = binaryImg.copy()
        row_projection = binaryImg.sum(axis=1)
        # 剪掉上下边框
        up_start = 0
        for i in range(len(row_projection)):
            if row_projection[i] > 50 * 255:
                up_start = i
                break
        down_end = 0
        for i in range(len(row_projection) - 1, 0, -1):
            if row_projection[i] > 50 * 255:
                down_end = i
                break
        binaryImg = binaryImg[up_start:down_end, :]
        return binaryImg

    def __detect_border(self, binaryImg):
        """
        竖直方向边界检测，处理结果用于字符分割
        :param binaryImg: 二值图
        :return: 边界信息
        """
        binaryImg = binaryImg.copy()
        column_projection = binaryImg.sum(axis=0) < 255 * 10
        # row_projection = binaryImg.sum(axis=1)
        borders = [0, 0]
        start = 0
        width = 0
        for end in range(1, len(column_projection)):
            if column_projection[end]:
                if end == len(column_projection) - 1:
                    borders.append(start)
                    borders.append(end)
                elif column_projection[start]:
                    width += 1
                else:
                    start = end
                    width = 1
            else:
                if column_projection[start] and width > 5:
                    borders.append(start)
                    borders.append(end)
                    width = 0
                    start = end
                else:
                    start = end
        borders.extend([len(column_projection)-1, len(column_projection)-1])
        return borders

    def __char_segment(self, binaryImage, borders):
        """
        字符分割
        :param binaryImage: 二值图
        :param borders: detect_border输出结果
        :return:
        """
        height, width = binaryImage.shape
        charImages = []
        for ind in range(len(borders)-1):
            if ind % 2 == 1:
                start = borders[ind]
                end = borders[ind + 1]
                if end - start > 5:
                    region = binaryImage[:, start:end+1]
                    region_row_projection = region.sum(axis=1)
                    # 根据在x（row）方向的投影筛选有效数字区域
                    if (region_row_projection > 0).sum() > 30:
                        # char "1"
                        if height / (end - start) > 4:
                            rows, cols = region.shape
                            pad = (rows // 2 - cols) // 2
                            region = np.hstack([np.zeros((rows, pad), dtype=region.dtype),
                                                region,
                                                np.zeros((rows, pad), dtype=region.dtype)])
                            pass
                        charImg = cv2.resize(region, (30, 60))
                        charImages.append(charImg)
        return charImages

    def getGrayImage(self):
        """
        灰度图
        :return:
        """
        return bgr2gray(self.bgrImage)

    def getBinaryImage(self):
        return gray2binary(self.getGrayImage())

    def getSegmentImage(self):
        """
        获取分割效果图
        :return:
        """
        binaryImage = self.getBinaryImage()
        # detect her borders
        borders = self.__detect_border(binaryImage)
        bgrImage = self.bgrImage.copy()
        for segLine in borders:
            bgrImage[:, segLine, :] = np.array([0, 0, 255])
        return bgrImage

    def getCharImages(self):
        """
        车牌图片转字符图片
        :param bgrImage:
        :return:
        """
        binaryImage = self.getBinaryImage()
        # crop vertical borders
        binaryImage = self.__crop_vertical_border(binaryImage)
        # detect her borders
        borders = self.__detect_border(binaryImage)
        # char segmentation
        charImages = self.__char_segment(binaryImage, borders)
        return charImages


def main():
    def test(fpath):
        bgrImage = cv2.imread(fpath, )
        if bgrImage is None:
            sys.exit(f"Failed to load image {fpath}!")
        splitter = CharSplitter(bgrImage)
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
    import os
    dirname = r'./dataset/车牌图片库'
    file_list = os.listdir(dirname)
    for fname in file_list:
        print(fname)
        if fname.endswith('bmp'):
            fpath = f'{dirname}/{fname}'
            test(fpath)


if __name__ == '__main__':
    main()

