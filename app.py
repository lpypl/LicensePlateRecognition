import sys
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QFileDialog, QMessageBox, QLabel, QSizePolicy
from PyQt5 import uic, QtCore
from PyQt5.QtGui import QIcon, QPalette, QPixmap, QImage


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
            'btnCharIdentify', 'imgLoad', 'imgSplit', 'imgChar0',
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

        self.openImageAndShow(filename, self.imgLoad)
        # TODO 获取图片存入self

    def openImageAndShow(self, filename, label):
        m_image = QImage(filename)
        if m_image.isNull():
            QMessageBox.information(self, "Image Viewer",
                                    "无法加载 %s." % filename)
            return
        lebalWidth = label.frameGeometry().width()
        lebalHeight = label.frameGeometry().height()
        print(lebalWidth, lebalHeight)
        m_image.scaled(lebalWidth,
                       lebalHeight,
                       aspectRatioMode=QtCore.Qt.KeepAspectRatio,
                       transformMode=QtCore.Qt.SmoothTransformation)
        label.setBackgroundRole(QPalette.Base)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        label.setScaledContents(True)
        label.setPixmap(QPixmap.fromImage(m_image))

    def btnSavePressed(self):
        #TODO what is this function does
        print("btnSavePressed")


app = QApplication(sys.argv)
window = Ui()
app.exec_()