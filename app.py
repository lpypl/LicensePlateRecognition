import sys
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QFileDialog, QMessageBox, QLabel, QSizePolicy
from PyQt5 import uic, QtCore
from PyQt5.QtGui import QIcon, QPalette, QPixmap, QImage
from pretreatment import *
from licenseLocator import *
from charSegment import *


class Ui(QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('app.ui', self)
        self.show()

        # 获取组件, 绑定点击事件
        widgets = [
            'btnOpen', 'btnSave', 'btnPictureGray', 'btnGrayScales',
            'btnMedianFilter', 'btnLicenceGray', 'btnLicenceBinary',
            'btnCharSplit', 'btnEdgeDetection', 'btnLocation',
            'btnCharIdentify', 'imgLoad', 'imgLocate', 'imgSplit', 'imgChar0',
            'imgChar1', 'imgChar2', 'imgChar3', 'imgChar4', 'imgChar5',
            'imgChar6',
        ]
        for widget in widgets:
            if widget.startswith('btn'):
                self[widget] = self.findChild(QPushButton, widget)
            elif widget.startswith('img'):
                self[widget] = self.findChild(QLabel, widget)

            if hasattr(self, (widget) + 'Pressed'):
                self[widget].clicked.connect(
                    getattr(self, widget + 'Pressed', None))

    def __setitem__(self, k, v):
        self.k = v

    def __getitem__(self, k):
        return self.k

    def __openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)",
            options=options)
        if fileName:
            print('__openFileNameDialog:' + fileName)
        return fileName

    def btnOpenPressed(self):
        filename = self.__openFileNameDialog()
        if not filename:
            return

        self.__openImageAndShow(filename, self.imgLoad)
        # TODO 获取图片存入self

    def __openImageAndShow(self, filename, label):
        m_image = QImage(filename)
        if m_image.isNull():
            QMessageBox.information(self, "Error",
                                    "无法加载 %s." % filename)
            return
        self.__showImageInLabel(m_image, label)

        bgrImage = cv2.imread(filename)

        self.pt = Pretreatment(bgrImage)
        self.ll = LicenseLocator(bgrImage)
        self.licenseImage = self.ll.getLicenseImage()
        if(self.licenseImage is not None):
            self.splitter = CharSplitter(self.licenseImage)

    def __showImageInLabel(self, m_image: QImage, label):
        if(m_image is None):
            print("__showImageInLabel m_image is None")
            return
        label.clear()

        lebalWidth = label.frameGeometry().width()
        lebalHeight = label.frameGeometry().height()
        print('label width and height:',lebalWidth, lebalHeight)
        m_image.scaled(lebalWidth,
                       lebalHeight,
                       aspectRatioMode=QtCore.Qt.KeepAspectRatio,
                       transformMode=QtCore.Qt.SmoothTransformation)
        label.setBackgroundRole(QPalette.Base)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        label.setScaledContents(True)
        label.setPixmap(QPixmap.fromImage(m_image))

    def btnPictureGrayPressed(self):
        if(hasattr(self, 'pt')):
            self.__showImageInLabel(self.CNp2QImage(
                self.pt.getGrayImage()), self.imgLoad)

    def btnGrayScalesPressed(self):
        if(hasattr(self, 'pt')):
            self.__showImageInLabel(self.CNp2QImage(
                self.pt.getHistEqImage()), self.imgLoad)

    def btnMedianFilterPressed(self):
        if(hasattr(self, 'pt')):
            self.__showImageInLabel(self.CNp2QImage(
                self.pt.getMedianImage()), self.imgLoad)

    def btnEdgeDetectionPressed(self):
        if(hasattr(self, 'll')):
            self.__showImageInLabel(self.CNp2QImage(
                self.ll.getRectImage()), self.imgLoad)

    def btnLocationPressed(self):
        if(hasattr(self, 'll')):
            self.__showImageInLabel(self.CNp2QImage(
                self.ll.getLicenseImage()), self.imgLocate)

    def CNp2QImage(self, image):
        """
            convert np image to QImage
        """
        if(image is None):
            print("CNp2QImage image is None")
            return None
        print('convert to QImage, (height width):', image.shape, type(image), image.data)

        data = image.data
        height, width, *channel = image.shape
        bytesPerLine = 3 * width
        if(isinstance(data, memoryview)):
            data = data.tobytes()

        if(len(channel) == 0):
            return QImage(data, width, height, width, QImage.Format_Grayscale8)
        elif(channel[0] == 3):
            # rgbSwapped显示bgr
            return QImage(data, width, height, width * 3, QImage.Format_RGB888).rgbSwapped()
        else:
            QMessageBox.information(self, "Error",
                                    "图片转换失败, channel必须是0或3。")

    def btnLicenceGrayPressed(self):
        if(hasattr(self, "splitter")):
            self.__showImageInLabel(self.CNp2QImage(
                self.splitter.getGrayImage()), self.imgLocate)

    def btnLicenceBinaryPressed(self):
        if(hasattr(self, "splitter")):
            self.__showImageInLabel(self.CNp2QImage(
                self.splitter.getBinaryImage()), self.imgLocate)

    def btnCharSplitPressed(self):
        if(hasattr(self, "splitter")):
            self.__showImageInLabel(self.CNp2QImage(
                self.splitter.getSegmentImage()), self.imgSplit)

            charImages = self.splitter.getCharImages()
            print(len(charImages))
            for n in range(0, 7):
                self.__showImageInLabel(self.CNp2QImage(
                    charImages[n]), getattr(self, 'imgChar' + str(n)))

    def btnSavePressed(self):
        # TODO what is this function does
        print("btnSavePressed")


def main():
    app = QApplication(sys.argv)
    window = Ui()
    app.exec_()


if __name__ == '__main__':
    main()
