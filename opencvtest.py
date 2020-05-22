import cv2
import sys
import numpy as np


def segment(fpath):
    img = cv2.imread(fpath)
    if img is None:
        sys.exit(f"Failed to load image {fpath}!")

    # output data
    print(img.shape)
    print(img[0, 0, :])

    # resize
    img = cv2.resize(img, (500, 130), cv2.INTER_AREA)

    # show & close
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # smooth
    img = cv2.blur(img, (3, 3))

    # binary
    rows, cols, channels = img.shape
    bimg = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            if img[r, c].min() > 130:
                bimg[r, c] = 255
            else:
                bimg[r, c] = 0
    cv2.imshow('W1', bimg, )
    cv2.moveWindow('W1', 0, 0)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 3x3 filter
    prewitt_x = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    prewitt_y = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    colorimg = img.copy()
    segimg = bimg.copy()
    boderimg = np.zeros((rows+2, cols+2))
    boderimg[1:-1, 1:-1] = segimg
    for r in range(rows):
        r += 1
        for c in range(cols):
            c += 1
            region = boderimg[r-1:r+2, c-1:c+2]
            # if np.abs(np.sum(region * prewitt_x)) > 5:
            #     colorimg[r-1, c-1] = np.array([0, 0, 255])
            if np.abs(np.sum(region * prewitt_y)) > 5:
                colorimg[r-1, c-1] = np.array([0, 0, 255])

    col_start = 0
    seglines = []
    for c in range(cols):
        if bimg[:, c].sum() > 255*5:
            col_end = c
            if col_end - col_start > 10:
                colorimg[:, (col_start + col_end)//2] = np.array([0, 255, 0])
                seglines.append((col_start + col_end)//2)
            col_start = c
    cv2.imshow('W2', colorimg)
    cv2.moveWindow('W2', 0, 300)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # subimg
    # segstart = seglines[0]
    # for segend in seglines[1:]:
    #     subimg = colorimg[:, segstart+1:segend, :]
    #     segstart = segend
    #     cv2.imshow('image', subimg)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


def main():
    import os
    dirname = r'/home/lpy/PythonProjects/LicensePlateRecognition/dataset/车牌图片库'
    file_list = os.listdir(dirname)
    for fname in file_list:
        print(fname)
        if fname.endswith('bmp'):
            fpath = f'{dirname}/{fname}'
            segment(fpath)


if __name__ == '__main__':
    main()

