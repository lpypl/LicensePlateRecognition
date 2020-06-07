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


def gray2binary(grayimg):
    return gray2binary_otsu(grayimg)


def image_filter(grayimg, method, scale=3):
    """
    中值/均值 平滑滤波
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


def gaussian_filter(img, K_size=3, sigma=1.3):
    """
    高斯平滑滤波
    :param img:
    :param K_size:
    :param sigma:
    :return:
    """
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()
    tmp = out.copy()
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    return out


def hist_equalization(img):
    """
    直方图均衡化
    :param img:
    :return:
    """
    img = img.copy()
    H, W = img.shape
    gray = np.zeros(256)
    for i in range(H):
        for j in range(W):
            gray[img[i][j]] += 1
    SumGray = np.zeros(256)
    SumGray = np.cumsum(gray)
    SumGray = np.array((SumGray * 255) / (H * W)).astype(np.int32)
    for i in range(H):
        for j in range(W):
            img[i][j] = SumGray[img[i][j]]
    return img
