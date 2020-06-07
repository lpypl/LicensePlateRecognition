import numpy as np


def bgr2gray(img):
    """
    BGR图像灰度化
    :param img: bgrImage
    :return: grayImage
    """
    # YCrCb
    return (img[:, :] * np.array([0.1140, 0.5870, 0.2990])).sum(axis=2).astype(np.uint8)


def gray2binary_otsu(grayimg):
    """
    OTSU灰度图像二值化
    :param grayimg:
    :return:
    """
    binaryImage = grayimg.copy()
    nbins = 256
    imhist, bins = np.histogram(grayimg.flatten(), np.arange(0, 257), normed=False)
    p = imhist / sum(imhist)
    threshold = 0
    sigma_max = 0

    for t in range(1, nbins):
        q_L = np.sum(p[0:t])
        q_H = np.sum(p[t:])
        if q_L == 0:
            miu_L = 0
        else:
            miu_L = np.sum(p[0:t] * np.arange(0, t, dtype=p.dtype)) / float(q_L)
        if q_H == 0:
            miu_H = 0
        else:
            miu_H = np.sum(p[t:] * np.arange(t, nbins, dtype=p.dtype)) / float(q_H)
        sigma_b = q_L * q_H * ((miu_L - miu_H) * (miu_L - miu_H))
        if sigma_b > sigma_max:
            sigma_max = sigma_b
            threshold = t
    # print(threshold)
    binaryImage[binaryImage >= threshold] = 255
    binaryImage[binaryImage < threshold] = 0
    return binaryImage


def image_filter(grayimg, method, scale=3):
    """
    平滑滤波
    :param grayimg: gray image
    :param method: median,mean
    :param scale: region scale
    :return:
    """
    rows, cols = grayimg.shape
    margin = scale // 2
    borderImg = np.zeros((rows + 2 * margin, cols + 2 * margin))
    borderImg[margin:-margin, margin:-margin] = grayimg
    outImage = np.zeros(grayimg.shape, grayimg.dtype)
    for r in range(rows):
        for c in range(cols):
            nr = r + margin
            nc = c + margin
            region = borderImg[nr - margin:nr + margin + 1, nc - margin:nc + margin + 1]
            pixels = region.flatten()
            pixels.sort()
            if method == "median":
                outImage[r, c] = pixels[margin + 1]
            elif method == "mean":
                outImage[r, c] = pixels.mean()
            else:
                raise ValueError(f"method类型错误")
    return outImage
