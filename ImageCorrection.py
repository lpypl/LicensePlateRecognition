def imageCorrection(bgrImage):
    import cv2
    import numpy as np

    originImage = bgrImage.copy()
    grayImage = cv2.cvtColor(originImage, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.medianBlur(grayImage, 3)
    # grayImage = cv2.medianBlur(grayImage, 3)
    grayImage = cv2.GaussianBlur(grayImage, (3, 3), 0)

    cannyImage = cv2.Canny(grayImage, 30, 100)
    # cv2.imshow('Canny', cannyImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    height, width = grayImage.shape
    lines = cv2.HoughLines(cannyImage, 1, 5 * np.pi / 180, 10)
    result = grayImage.copy()
    maxLengthLineH = np.array([0.0, 0.0])
    maxLengthLineV = np.array([0.0, 0.0])

    if lines is None:
        return originImage
    for line in lines[0]:
        rho = line[0]  # 第一个元素是距离rho
        theta = line[1]  # 第二个元素是角度theta
        print(rho)
        print(theta)
        # if np.abs(rho) < 26:
        #     continue
        if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
            print('Vertical')
            # 该直线与第一行的交点
            pt1 = (int(rho / np.cos(theta)), 0)
            # 该直线与最后一行的焦点
            pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
            # 绘制一条白线
            cv2.line(result, pt1, pt2, 255)

            if np.abs(rho) > np.abs(maxLengthLineV[0]):
                maxLengthLineV = line
        else:  # 水平直线
            print('Horizontal')
            # 该直线与第一列的交点
            pt1 = (0, int(rho / np.sin(theta)))
            # 该直线与最后一列的交点
            pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
            # 绘制一条直线
            cv2.line(result, pt1, pt2, 255, 1)

            if np.abs(rho) > np.abs(maxLengthLineH[0]):
                maxLengthLineH = line

    cv2.imshow('Result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    h, w = grayImage.shape[:2]
    center = (w // 2, h // 2)
    corrImage = originImage.copy()
    if maxLengthLineH[0] > 0:
        print('length>0')
        M = cv2.getRotationMatrix2D(center,  180 * (maxLengthLineH[1]) / np.pi - 90, 1)
    elif maxLengthLineH[0] < 0:
        print('length<=0')
        M = cv2.getRotationMatrix2D(center, 180 * (maxLengthLineH[1]) / np.pi - 90, 1)
    else:
        return originImage

    corrImage = cv2.warpAffine(corrImage, M, (w, h))
    # cv2.imshow('Result', corrImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return corrImage
