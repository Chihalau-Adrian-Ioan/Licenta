# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'InterfataAplicatie.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(924, 532)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.menuLayout = QtWidgets.QVBoxLayout()
        self.menuLayout.setObjectName("menuLayout")
        self.commandsLayout = QtWidgets.QGridLayout()
        self.commandsLayout.setObjectName("commandsLayout")
        self.droneWattPerKmLabel = QtWidgets.QLabel(self.centralwidget)
        self.droneWattPerKmLabel.setEnabled(False)
        self.droneWattPerKmLabel.setObjectName("droneWattPerKmLabel")
        self.commandsLayout.addWidget(self.droneWattPerKmLabel, 2, 0, 1, 1)
        self.locChooseLabel = QtWidgets.QLabel(self.centralwidget)
        self.locChooseLabel.setEnabled(False)
        self.locChooseLabel.setObjectName("locChooseLabel")
        self.commandsLayout.addWidget(self.locChooseLabel, 1, 0, 1, 1)
        self.locGenCourseButton = QtWidgets.QPushButton(self.centralwidget)
        self.locGenCourseButton.setEnabled(False)
        self.locGenCourseButton.setObjectName("locGenCourseButton")
        self.commandsLayout.addWidget(self.locGenCourseButton, 3, 0, 1, 2)
        self.loadMapButton = QtWidgets.QPushButton(self.centralwidget)
        self.loadMapButton.setObjectName("loadMapButton")
        self.commandsLayout.addWidget(self.loadMapButton, 0, 1, 1, 1)
        self.droneWattPerKmSpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.droneWattPerKmSpinBox.setEnabled(False)
        self.droneWattPerKmSpinBox.setProperty("value", 10)
        self.droneWattPerKmSpinBox.setObjectName("droneWattPerKmSpinBox")
        self.commandsLayout.addWidget(self.droneWattPerKmSpinBox, 2, 1, 1, 1)
        self.mapStatusLabel = QtWidgets.QLabel(self.centralwidget)
        self.mapStatusLabel.setObjectName("mapStatusLabel")
        self.commandsLayout.addWidget(self.mapStatusLabel, 0, 0, 1, 1)
        self.locChooseComboBox = QtWidgets.QComboBox(self.centralwidget)
        self.locChooseComboBox.setEnabled(False)
        self.locChooseComboBox.setObjectName("locChooseComboBox")
        self.commandsLayout.addWidget(self.locChooseComboBox, 1, 1, 1, 1)
        self.menuLayout.addLayout(self.commandsLayout)
        self.tourInfoGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.tourInfoGroupBox.setEnabled(False)
        self.tourInfoGroupBox.setObjectName("tourInfoGroupBox")
        self.formLayout_3 = QtWidgets.QFormLayout(self.tourInfoGroupBox)
        self.formLayout_3.setObjectName("formLayout_3")
        self.consumedPowerWLabel = QtWidgets.QLabel(self.tourInfoGroupBox)
        self.consumedPowerWLabel.setObjectName("consumedPowerWLabel")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.consumedPowerWLabel)
        self.consumedPowerWLineEdit = QtWidgets.QLineEdit(self.tourInfoGroupBox)
        self.consumedPowerWLineEdit.setEnabled(False)
        self.consumedPowerWLineEdit.setReadOnly(True)
        self.consumedPowerWLineEdit.setObjectName("consumedPowerWLineEdit")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.consumedPowerWLineEdit)
        self.tourTotalDistKmLabel = QtWidgets.QLabel(self.tourInfoGroupBox)
        self.tourTotalDistKmLabel.setObjectName("tourTotalDistKmLabel")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.tourTotalDistKmLabel)
        self.tourTotalDistKmLineEdit = QtWidgets.QLineEdit(self.tourInfoGroupBox)
        self.tourTotalDistKmLineEdit.setEnabled(False)
        self.tourTotalDistKmLineEdit.setReadOnly(True)
        self.tourTotalDistKmLineEdit.setObjectName("tourTotalDistKmLineEdit")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.tourTotalDistKmLineEdit)
        self.currentDistanceCrossedKmLabel = QtWidgets.QLabel(self.tourInfoGroupBox)
        self.currentDistanceCrossedKmLabel.setObjectName("currentDistanceCrossedKmLabel")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.currentDistanceCrossedKmLabel)
        self.currentDistanceCrossedKmLineEdit = QtWidgets.QLineEdit(self.tourInfoGroupBox)
        self.currentDistanceCrossedKmLineEdit.setEnabled(False)
        self.currentDistanceCrossedKmLineEdit.setReadOnly(True)
        self.currentDistanceCrossedKmLineEdit.setObjectName("currentDistanceCrossedKmLineEdit")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.currentDistanceCrossedKmLineEdit)
        self.totalStreetsLengthKmLabel = QtWidgets.QLabel(self.tourInfoGroupBox)
        self.totalStreetsLengthKmLabel.setObjectName("totalStreetsLengthKmLabel")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.totalStreetsLengthKmLabel)
        self.totalStreetsLengthKmLineEdit = QtWidgets.QLineEdit(self.tourInfoGroupBox)
        self.totalStreetsLengthKmLineEdit.setEnabled(False)
        self.totalStreetsLengthKmLineEdit.setReadOnly(True)
        self.totalStreetsLengthKmLineEdit.setObjectName("totalStreetsLengthKmLineEdit")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.totalStreetsLengthKmLineEdit)
        self.additionalDistanceKmLabel = QtWidgets.QLabel(self.tourInfoGroupBox)
        self.additionalDistanceKmLabel.setObjectName("additionalDistanceKmLabel")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.additionalDistanceKmLabel)
        self.additionalDistanceKmLineEdit = QtWidgets.QLineEdit(self.tourInfoGroupBox)
        self.additionalDistanceKmLineEdit.setEnabled(False)
        self.additionalDistanceKmLineEdit.setReadOnly(True)
        self.additionalDistanceKmLineEdit.setObjectName("additionalDistanceKmLineEdit")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.additionalDistanceKmLineEdit)
        self.menuLayout.addWidget(self.tourInfoGroupBox)
        self.horizontalLayout.addLayout(self.menuLayout)
        self.mapLayout = QtWidgets.QVBoxLayout()
        self.mapLayout.setObjectName("mapLayout")
        self.horizontalLayout.addLayout(self.mapLayout)
        self.horizontalLayout.setStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 924, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Simulator traseu zonă rezidențială dronă"))
        self.droneWattPerKmLabel.setText(_translate("MainWindow", "Consum putere dronă (W/km)"))
        self.locChooseLabel.setText(_translate("MainWindow", "Alege localitate"))
        self.locGenCourseButton.setText(_translate("MainWindow", "Generează tur"))
        self.loadMapButton.setText(_translate("MainWindow", "Încarcă harta..."))
        self.mapStatusLabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; "
                                                             "color:#aa0000;\">Status: Hartă "
                                                             "neîncărcată</span></p></body></html>"))
        self.tourInfoGroupBox.setTitle(_translate("MainWindow", "Informații Tur"))
        self.consumedPowerWLabel.setText(_translate("MainWindow", "Putere consumată curentă (W)"))
        self.tourTotalDistKmLabel.setText(_translate("MainWindow", "Distanță totală tur (km)"))
        self.currentDistanceCrossedKmLabel.setText(_translate("MainWindow", "Distanță parcursă dronă curentă (km)"))
        self.totalStreetsLengthKmLabel.setText(_translate("MainWindow", "Lungime totală străzi (km)"))
        self.additionalDistanceKmLabel.setText(_translate("MainWindow", "Distanță parcursă suplimentar (km)"))
