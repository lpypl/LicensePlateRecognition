import cv2
import sys
import numpy as np


def img2gray(img):
    # YCrCb
    return (img[:, :] * np.array([0.1140, 0.5870, 0.2990])).sum(axis=2).astype(np.uint8)


def gray2binary(grayimg, threshold=185):
    # rows, cols = grayimg.shape
    # bimg = np.zeros((rows, cols))
    # for r in range(rows):
    #     for c in range(cols):
    #         if grayimg[r, c] > threshold:
    #             bimg[r, c] = 255
    #         else:
    #             bimg[r, c] = 0
    thres, bimg = cv2.threshold(grayimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return bimg


# def edgeDetection(grayimg):
#     rows, cols = grayimg.shape
#     # 3x3 filter
#     prewitt_x = np.array([
#         [-1, -1, -1],
#         [0, 0, 0],
#         [1, 1, 1]
#     ])
#     prewitt_y = np.array([
#         [-1, 0, 1],
#         [-1, 0, 1],
#         [-1, 0, 1]
#     ])
#     boderimg = np.zeros((rows+2, cols+2))
#     boderimg[1:-1, 1:-1] = grayimg
#     x_edges = []
#     y_edges = []
#
#     for r in range(rows):
#         r += 1
#         for c in range(cols):
#             c += 1
#             region = boderimg[r-1:r+2, c-1:c+2]
#             if np.abs(np.sum(region * prewitt_x)) > 300:
#                 x_edges.append((r-1, c-1))
#             if np.abs(np.sum(region * prewitt_y)) > 300:
#                 y_edges.append((r-1, c-1))
#     return x_edges, y_edges


def segment(bimg):
    rows, cols = bimg.shape
    col_start = 0
    seglines = []
    for c in range(cols):
        if bimg[:, c].sum() > 255*5:
            col_end = c
            if col_end - col_start > 8:
                seglines.append((col_start + col_end)//2)
            col_start = c
    if len(seglines) > 0:
        if bimg[:, 0:seglines[0]].sum() / (seglines[0] * rows * 255) < 1/20:
            del(seglines[0])
    if len(seglines) > 0:
        if bimg[:, seglines[-1]:].sum() / ((cols - seglines[-1]) * rows * 255) < 1/20:
            del(seglines[-1])
    seglines = [0] + seglines + [cols-1]
    return seglines


def get_digits(img, seglines, fpath):
    seglines = seglines[-6:]
    digits = []
    labels = list(fpath.split('/')[-1][:-4])
    for i in range(len(seglines) - 1):
        digit = img[:, seglines[i]:seglines[i+1]+1, :]
        digit = cv2.resize(digit, (70, 130))
        digits.append(digit)

    return digits, labels


def save_digits(digits, labels, prefix='./digits'):
    license = ''.join(labels)
    for i in range(len(digits)):
        fname = f'{labels[i]}_{license}_{i}'
        cv2.imwrite(f'{prefix}/{fname}.bmp', digits[i])


def img_enhance_smooth(grayimg):
    grayimg = grayimg.copy()
    # 进行直方图均衡化会导致大量噪点，直接使用效果很好
    # histogram equalization
    # grayimg = cv2.equalizeHist(grayimg)
    # smooth
    for siz in [3, 5]:
        grayimg = cv2.medianBlur(grayimg, siz)
        grayimg = cv2.blur(grayimg, (siz, siz))
    return grayimg


def execute(fpath):
    img = cv2.imread(fpath, )
    if img is None:
        sys.exit(f"Failed to load image {fpath}!")
    # resize
    img = cv2.resize(img, (500, 130), cv2.INTER_AREA)
    # to gray
    grayimg = img2gray(img)
    smooth_grayimg = img_enhance_smooth(grayimg)
    bimg = gray2binary(smooth_grayimg)
    seglines = segment(bimg)

    smooth_grayimg = img_enhance_smooth(grayimg)
    bimg = gray2binary(smooth_grayimg, 175)
    for seg in seglines:
        bimg[:, seg] = 255

    digits, labels = get_digits(img, seglines, fpath)
    save_digits(digits, labels)

    cv2.imshow('W1', bimg, )
    cv2.moveWindow('W1', 0, 0)
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

