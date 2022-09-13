# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'loadingScreen.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QMovie


class LoadingScreen(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setModal(True)
        self.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)
        self.loadingLabel = QtWidgets.QLabel(self)
        self.notificationLabel = QtWidgets.QLabel(self)
        self.movie = QMovie("Spinner-1s-108px.gif")

    def setupUi(self):
        self.setObjectName("Form")
        self.setFixedSize(400, 261)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        self.notificationLabel.setGeometry(QtCore.QRect(60, 10, 291, 91))
        self.notificationLabel.setObjectName("notificationLabel")

        self.loadingLabel.setGeometry(QtCore.QRect(150, 100, 111, 121))
        self.loadingLabel.setCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        self.loadingLabel.setText("")
        self.loadingLabel.setObjectName("loadingLabel")
        self.loadingLabel.setMovie(self.movie)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def startAnimation(self):
        self.movie.start()
        self.show()

    def stopAnimation(self):
        self.movie.stop()
        self.close()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Form", "Form"))
        self.notificationLabel.setText(_translate("Form",
                                                  "<html><head/><body><p align=\"center\"><br/></p></body></html>"))


def quitApp(app, ui):
    ui.stopAnimation()
    app.quit()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    ui = LoadingScreen()
    ui.setupUi()
    ui.show()
    ui.startAnimation()
    QTimer.singleShot(2000, lambda: quitApp(app, ui))
    sys.exit(app.exec_())