import cv2
import sys
from basicAlgorithm import *


class CharSplitter:
    def __init__(self, licenseBGRImage):
        # resize
        self.bgrImage = cv2.resize(licenseBGRImage, (500, 130), cv2.INTER_AREA)
        self.cropBgrImage = self.bgrImage.copy()

    def __crop_vertical_border(self, binaryImg):
        """
        借助在Y轴上的投影裁减掉上方和下方的边界
        :param binaryImg: 二值图像
        :return: 裁减掉上方和下方的边界的二值图像
        """
        height, width = binaryImg.shape
        binaryImg = binaryImg.copy()
        row_projection = binaryImg.sum(axis=1)
        row_threshold = 100
        # 剪掉上下边框
        up_start = 0
        for i in range(len(row_projection)):
            if row_threshold * 255 < row_projection[i] < (width - row_threshold) * 255:
                up_start = i
                break
        down_end = 0
        for i in range(len(row_projection) - 1, 0, -1):
            if row_threshold * 255 < row_projection[i] < (width - row_threshold) * 255:
                down_end = i
                break

        binaryImg = binaryImg[up_start:down_end, :]
        self.cropBgrImage = self.bgrImage[up_start:down_end, :]

        height, width = binaryImg.shape
        col_projection = binaryImg.sum(axis=0)
        left_start = 0
        for i in range(len(col_projection)):
            if int(height * 0.2) * 255 < col_projection[i] < int(height * 0.7) * 255:
                left_start = i
                break
        right_end = 0
        for i in range(len(col_projection) - 1, 0, -1):
            if int(height * 0.2) * 255 < col_projection[i] < int(height * 0.7) * 255:
                right_end = i
                break
        binaryImg = binaryImg[:, left_start:right_end]
        self.cropBgrImage = self.cropBgrImage[:, left_start:right_end]
        return binaryImg

    def __detect_border(self, binaryImg):
        """
        竖直方向边界检测，处理结果用于字符分割
        :param binaryImg: 二值图
        :return: 边界信息
        """
        binaryImg = binaryImg.copy()
        # detect
        height, width = binaryImg.shape
        # 列元素少于一定比例，认为是空列
        column_projection = binaryImg.sum(axis=0) < 255 * int(height * 0.07)
        # row_projection = binaryImg.sum(axis=1)
        borders = []
        borders = [0, 0]
        start = 0
        spaceWidth = 0
        for end in range(1, len(column_projection)):
            if column_projection[end]:
                if end == len(column_projection) - 1:
                    borders.append(start)
                    borders.append(end)
                elif column_projection[start]:
                    spaceWidth += 1
                else:
                    start = end
                    spaceWidth = 1
            else:
                # 含有效像素比例低于X，认为是空白区域(边界)
                if column_projection[start] and spaceWidth < width*0.1 and binaryImg[:,
                                                start:start + spaceWidth].sum() < spaceWidth * height * 255 * 0.10:
                    # if column_projection[start] and spaceWidth > int(width * 0.01):
                    borders.append(start)
                    borders.append(end)
                    spaceWidth = 0
                    start = end
                else:
                    start = end
        borders.extend([len(column_projection) - 1, len(column_projection) - 1])
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
        for ind in range(len(borders) - 1):
            if ind % 2 == 1:
                start = borders[ind]
                end = borders[ind + 1]
                if end - start > 10:
                    region = binaryImage[:, start:end + 1]
                    region_row_projection = region.sum(axis=1)
                    # 根据在x（row）方向的投影筛选有效数字区域
                    if (region_row_projection > 0).sum() / height > 0.3:
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
        grayImage = bgr2gray(self.bgrImage)
        # return grayImage
        smoothImage = cv2.medianBlur(grayImage, 3)
        # smoothImage = image_filter(grayImage, 'median', 3)
        # smoothImage = image_filter(smoothImage, 'mean', 3)
        # smoothImage = cv2.GaussianBlur(smoothImage, (3, 3), 0)
        # smoothImage = cv2.GaussianBlur(smoothImage, (3, 3), 0.8)
        return smoothImage

    def getBinaryImage(self):
        # crop vertical borders
        binaryImg = gray2binary(self.getGrayImage())
        binaryImg = self.__crop_vertical_border(binaryImg)
        binaryImg = cv2.resize(binaryImg, (500, 100))
        self.cropBgrImage = cv2.resize(self.cropBgrImage, (500, 100))
        return binaryImg

    def getSegmentImage(self):
        """
        获取分割效果图
        :return:
        """
        binaryImage = self.getBinaryImage()
        # detect her borders
        borders = self.__detect_border(binaryImage)
        bgrImage = self.cropBgrImage.copy()
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
        # # crop vertical borders
        # binaryImage = self.__crop_vertical_border(binaryImage)
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
