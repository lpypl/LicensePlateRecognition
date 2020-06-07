import cv2
import sys
from basicAlgorithm import *


def smooth_filter(grayimg):
    """
    平滑滤波
    :param grayimg:
    :return: smooth_grayimg
    """
    grayimg = grayimg.copy()
    # 进行直方图均衡化会导致大量噪点，直接使用效果很好
    # grayimg = cv2.equalizeHist(grayimg)
    # smooth
    for siz in [3]:
        # todo 实现中值滤波和均值滤波
        # grayimg = cv2.medianBlur(grayimg, siz)
        grayimg = image_filter(grayimg, "median", siz)
        grayimg = image_filter(grayimg, "mean", siz)
        # grayimg = cv2.blur(grayimg, (siz, siz))
    return grayimg


def crop_vertical_border(binaryImg):
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


def detect_border(binaryImg):
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


def char_segment(binaryImage, borders):
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


def license2charImg(bgrImage):
    """
    车牌识别
    :param bgrImage:
    :return:
    """
    # resize
    bgrImage = cv2.resize(bgrImage, (500, 130), cv2.INTER_AREA)
    # to gray
    grayimg = bgr2gray(bgrImage)
    smooth_grayimg = smooth_filter(grayimg)
    binaryImage = gray2binary_otsu(smooth_grayimg)
    cv2.imshow('W1', binaryImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # crop vertical borders
    binaryImage = crop_vertical_border(binaryImage)
    # detect her borders
    borders = detect_border(binaryImage)
    # char segmentation
    charImages = char_segment(binaryImage, borders)
    return charImages


def execute(fpath):
    bgrImage = cv2.imread(fpath, )
    if bgrImage is None:
        sys.exit(f"Failed to load image {fpath}!")

    charImages = license2charImg(bgrImage)
    imgs = np.hstack(charImages)
    cv2.imshow('W1', imgs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    import os
    dirname = r'/home/lpy/PythonProjects/LicensePlateRecognition/dataset/车牌图片库'
    file_list = os.listdir(dirname)
    for fname in file_list:
        print(fname)
        if fname.endswith('bmp'):
            fpath = f'{dirname}/{fname}'
            execute(fpath)


if __name__ == '__main__':
    main()

