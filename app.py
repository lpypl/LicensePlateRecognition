import sys
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QFileDialog, QMessageBox, QLabel, QSizePolicy, QTextEdit
from PyQt5 import uic, QtCore
from PyQt5.QtGui import QIcon, QPalette, QPixmap, QImage
from pretreatment import *
from licenseLocator import *
from charSegment import *
from licenseReco import LicenseReco


class Ui(QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('app.ui', self)
        self.show()

        # 用于保存当前步奏得到的图片
        self.c_pic = None
        # 获取组件, 绑定点击事件
        widgets = [
            'btnOpen', 'btnSave', 'btnPictureGray', 'btnGrayScales',
            'btnMedianFilter', 'btnLicenceGray', 'btnLicenceBinary',
            'btnCharSplit', 'btnEdgeDetection', 'btnLocation',
            'btnCharIdentify', 'imgLoad', 'imgLocate', 'imgSplit', 'imgChar0',
            'imgChar1', 'imgChar2', 'imgChar3', 'imgChar4', 'imgChar5',
            'imgChar6', 'txtResult'
        ]
        for widget in widgets:
            if widget.startswith('btn'):
                self[widget] = self.findChild(QPushButton, widget)
            elif widget.startswith('img'):
                self[widget] = self.findChild(QLabel, widget)
            elif widget.startswith('txt'):
                self[widget] = self.findChild(QTextEdit, widget)

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

    def __saveFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "QFileDialog.getSaveFileName()",
            "save.jpg",
            "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)",
            options=options)
        if fileName:
            print('__openFileNameDialog:' + fileName)
        return fileName

    def btnOpenPressed(self):
        filename = self.__openFileNameDialog()
        if not filename:
            return
        self.clearAll()
        self.__openImageAndShow(filename, self.imgLoad)

    def __openImageAndShow(self, filename, label):
        m_image = QImage(filename)
        if m_image.isNull():
            QMessageBox.information(self, "Error",
                                    "无法加载 %s." % filename)
            return
        self.__showImageInLabel(m_image, label)

        # win10 环境下cv2.imread 如果路径出现中文会返回None
        # bgrImage = cv2.imread(filename)
        import PIL
        bgrImage = cv2.cvtColor(np.array(PIL.Image.open(filename).convert('RGB')), cv2.COLOR_RGB2BGR)

        self.pt = Pretreatment(bgrImage)
        self.ll = LicenseLocator(bgrImage)
        self.licenseImage = self.ll.getLicenseImage()
        if(self.licenseImage is not None):
            self.splitter = CharSplitter(self.licenseImage)
            self.reco = LicenseReco(digit_model_path='./CNNCharReco/digit_model.pkl',
                                    han_model_path='./CNNCharReco/han_model.pkl')

    def __showImageInLabel(self, m_image: QImage, label:QLabel):
        if(m_image is None):
            print("__showImageInLabel m_image is None")
            return
        label.clear()

        self.c_pic = m_image

        lebalWidth = label.frameGeometry().width()
        lebalHeight = label.frameGeometry().height()
        print('label width and height:', lebalWidth, lebalHeight)
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
        print('convert to QImage, (height width):',
              image.shape, type(image), image.data)

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
            for n in range(0, 7 if len(charImages) > 7 else len(charImages)):
                self.__showImageInLabel(self.CNp2QImage(
                    charImages[n]), getattr(self, 'imgChar' + str(n)))

    def btnCharIdentifyPressed(self):
        if(hasattr(self, "splitter")):
            result = self.reco.predict(self.splitter.getCharImages())
            self.txtResult.setPlainText(result)

    def btnSavePressed(self):
        filename = self.__saveFileNameDialog()
        if filename is not None:
            if not self.c_pic.save(filename):
                QMessageBox.information(self, "Error",
                                        "图片保存失败")
    def clearAll(self):
        """
        'imgLocate', 'imgSplit', 'imgChar0',
        'imgChar1', 'imgChar2', 'imgChar3', 'imgChar4', 'imgChar5',
        'imgChar6', 'txtResult'
        """
        widgets = ['imgLocate', 'imgSplit', 'imgChar0',
        'imgChar1', 'imgChar2', 'imgChar3', 'imgChar4', 'imgChar5',
        'imgChar6']
        for widget in widgets:
            getattr(self, widget, None).clear()

        self.txtResult.setPlainText("")


def main():
    app = QApplication(sys.argv)
    window = Ui()
    app.exec_()


if __name__ == '__main__':
    main()
