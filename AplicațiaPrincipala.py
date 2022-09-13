import sys
import threading
import time
import xml.etree.ElementTree as ET
from threading import Thread

import networkx as nx
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from InterfataAplicatie import Ui_MainWindow
from LoadingScreen import LoadingScreen
from XMLParser import generate_course


class InterfataAplicatie(QMainWindow):
    def __init__(self):
        super().__init__()
        self.eventHandlerThread = threading.Thread()  # thread de tratare al event-urilor din XMLParser
        self._stopThreads = False   # boolean pentru oprirea tuturor thead-urilor active când se închide aplicatia
        self._selectedFilename = None  # fisierul selectat din fereastra de selecție al fișierelor tip .osm
        self.solution_content = {}  # rezultatul funcției generateCourse din XMLParser

        self.ui = Ui_MainWindow()  # UI-ul ferestrei
        self.ui.setupUi(self)

        # asignarea funcțiilor atunci când butoanele sunt apăsate
        self.ui.loadMapButton.clicked.connect(self.loadMap)
        self.ui.locGenCourseButton.clicked.connect(self.generateCourse)

        # fereastra de încărcare din timpul procesării turului eulerian
        self.loadingScreen = LoadingScreen()
        self.loadingScreen.setupUi()

        self.figure = Figure(figsize=(10, 5))  # figura pe care se va desena grafuri

        self.mapCanvas = FigureCanvas(self.figure)  # plansa pe care se va desena graful
        self.ax = self.mapCanvas.figure.subplots()  # axa care contribuie la desenarea grafurilor în figura din canvas
        self.ui.mapLayout.addWidget(self.mapCanvas)
        self.ui.mapLayout.addWidget(NavigationToolbar(self.mapCanvas, self))  # adaugarea barei de navigatie pentru graf

        self.show()

    # se suprascrie metoda de închidere al aplicației, pentru a opri mai întâi toate thread-urile active
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self._stopThreads = True
        if self.eventHandlerThread.is_alive():
            self.eventHandlerThread.join()
        a0.accept()

    # metoda de pornire al animației turului eulerian pe graful orasului selectat
    def startCourseAnimation(self):
        print("Started animation! ;)")

        # se extrag mai intai datele din solutia obtinută
        graph, solution = self.solution_content['graph'], self.solution_content['solution']
        total_streets_length, additional_distance, total_tour_length, current_drone_dist_list = \
            self.solution_content['lungime_totala_strazi'], self.solution_content['distanta_parcursa_suplimentar'],\
            self.solution_content['distanta_totala_tur'], self.solution_content['distanta parcursa drona curenta']
        drone_consumption = self.ui.droneWattPerKmSpinBox.value()

        # sunt setate culorile nodurilor și muchiilor ce vor fi vizitate de-a lungul animației
        node_colors = {node: 'gray' for node in graph.nodes}
        edge_colors = {(edge[0], edge[1]): 'gray' for edge in graph.edges}
        node_pos = {node[0]: (node[1]['lat'], node[1]['lon']) for node in graph.nodes(data=True)}
        visited_edges_colors = ['gray', 'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']

        # funcția de inițializare al animației (pregătirea planșei și al grafului, inițial nevizitat)
        def init_animation():
            self.ui.totalStreetsLengthKmLineEdit.setText(f'{total_streets_length} km')
            self.ui.additionalDistanceKmLineEdit.setText(f'{additional_distance} km')
            self.ui.tourTotalDistKmLineEdit.setText(f'{total_tour_length} km')
            self.ui.currentDistanceCrossedKmLineEdit.setText(f'{current_drone_dist_list[0]} km')
            self.ui.consumedPowerWLineEdit.setText(f'{current_drone_dist_list[0] * drone_consumption} W')
            self.ax.cla()
            self.mapCanvas.figure.suptitle(f'Graful strazilor detectate din {self.ui.locChooseComboBox.currentText()} '
                                           f'(conform hartii {self._selectedFilename})')
            self.ax.set_facecolor("#E5FFE5")
            nx.draw_networkx(graph, node_color='gray', edge_color='gray', pos=node_pos, node_size=5,
                             with_labels=False, ax=self.ax)
            for node_index in range(len(node_colors)):
                if node_index == solution[0]:
                    node_colors[node_index] = 'yellow'
                    break
            nx.draw_networkx_nodes(graph, pos=node_pos, nodelist=[solution[0]],
                                   node_color=node_colors[solution[0]],
                                   node_size=5, ax=self.ax)
            self.mapCanvas.draw()

        # desenarea unui frame al animației
        # culoarea muchiei se va schimba în funcție de câte ori va fi vizitată
        def animation_frame(i):
            self.ui.currentDistanceCrossedKmLineEdit.setText(f'{current_drone_dist_list[i+1]} km')
            self.ui.consumedPowerWLineEdit\
                .setText(f'{"{:.2f}".format(current_drone_dist_list[i + 1] * drone_consumption)} W')
            current_node = solution[i]
            next_node = solution[i + 1]
            if current_node == solution[0]:
                node_colors[current_node] = '#FF007F'
            else:
                node_colors[current_node] = 'white'
            node_colors[next_node] = 'black'

            if (current_node, next_node) in edge_colors:
                current_edge = (current_node, next_node)
            else:
                current_edge = (next_node, current_node)
            for color_index in range(len(visited_edges_colors) - 1):
                if visited_edges_colors[color_index] == edge_colors[current_edge]:
                    edge_colors[current_edge] = visited_edges_colors[color_index + 1]
                    break
            nx.draw_networkx_nodes(graph, pos=node_pos, node_color=[node_colors[current_node], node_colors[next_node]],
                                   nodelist=[current_node, next_node], node_size=5, ax=self.ax)
            nx.draw_networkx_edges(graph, pos=node_pos, edgelist=[current_edge], edge_color=edge_colors[current_edge],
                                   ax=self.ax)
            self.mapCanvas.draw()

        # procedeul de redare al animației
        init_animation()
        for index in range(len(solution) - 1):
            # dacă s-a oferit semnalul de stop de la închiderea aplicației, se oprește animația
            if self._stopThreads:
                break
            # altfel se oferă un timp de 0.3 secunde între fiecare frame al animației și apoi se redă frame-ul curent
            time.sleep(0.3)
            animation_frame(index)

    # metoda de generare al soluției, apelat la apăsarea butonului de generare tur
    def generateCourse(self):
        selectedSettlement = self.ui.locChooseComboBox.currentText()
        if selectedSettlement == '':
            msg = QMessageBox()
            msg.setWindowTitle("Warning!")
            msg.setText("No settlement selected or none found on map!")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()
        else:
            # se oprește thread-ul de event-uri dacă este deja activ de la o prelucrare anterioară
            if self.eventHandlerThread.is_alive():
                self._stopThreads = True
                self.eventHandlerThread.join()
                self._stopThreads = False

            # se setează event-urile pentru etapele generării soluției
            finishSolutionEvent = threading.Event()
            roadExtractionEvent = threading.Event()
            routeProcessingEvent = threading.Event()
            animationProcessingEvent = threading.Event()
            print(f'Selected Settlement: {selectedSettlement}')

            # după se pornește firul de execuție al generării soluției
            processingThread = Thread(target=lambda: generate_course(self._selectedFilename, selectedSettlement,
                                                                     roadExtractionEvent, routeProcessingEvent,
                                                                     animationProcessingEvent, finishSolutionEvent,
                                                                     self.solution_content))
            processingThread.start()

            # in timpul acesta, se porneste fereastra de incarcare, care blocheaza interfata principala
            self.loadingScreen = LoadingScreen()
            self.loadingScreen.setupUi()
            self.loadingScreen.startAnimation()

            # se porneste manager-ul de event-uri, care schimba textul din loading screeen
            self.eventHandlerThread = Thread(target=self.handleEvents, args=[roadExtractionEvent, routeProcessingEvent,
                                                                             animationProcessingEvent,
                                                                             finishSolutionEvent])
            self.eventHandlerThread.start()

    # metoda de gestionare al event-uriolor din functia de generare al solutiei
    # in functie de în ce etapa se afla, se schimba textul loading screen-ului
    # odata finalizat, se incepe animatia grafului
    def handleEvents(self, roadExtraction: threading.Event, routeProcessing: threading.Event,
                     animationProcessing: threading.Event, stop: threading.Event):
        if roadExtraction.wait():
            self.loadingScreen.notificationLabel.setText('Extragere străzi și intersecții localitate...')
        if routeProcessing.wait():
            self.loadingScreen.notificationLabel.setText('Calculare tur eulerian localitate...')
        if animationProcessing.wait():
            self.loadingScreen.notificationLabel.setText('Creare animație tur...')
        if stop.wait():
            self.loadingScreen.notificationLabel.setText('Gata!')
            self.loadingScreen.stopAnimation()
            self.ui.tourInfoGroupBox.setEnabled(True)
            self.startCourseAnimation()

    # metoda de incarcare al oraselor si a harti spre prelucrarea acestora in functia de generare al solutiei
    def loadMap(self):
        self._selectedFilename, _ = QFileDialog.getOpenFileName(self, "Open Image",
                                                                filter="OpenStreetMap Files (*.osm)")
        print(self._selectedFilename)
        if self._selectedFilename != '':
            tree = ET.parse(self._selectedFilename)
            root = tree.getroot()
            settlements_list = []

            # gasirea oraselor se face pe baza etichetelor strazilor care le apartin
            for way in root.iter('way'):
                name = None
                highway_type = None
                for tag in way.iter('tag'):
                    if tag.attrib['k'] == 'highway' and tag.attrib['v'] in ['primary', 'secondary', 'tertiary',
                                                                            'residential']:
                        highway_type = tag.attrib['v']
                    if tag.attrib['k'] == 'is_in:city':
                        name = tag.attrib['v']
                if highway_type is not None and name is not None and name not in settlements_list:
                    settlements_list.append(name)

            print(settlements_list)

            _translate = QtCore.QCoreApplication.translate
            self.ui.mapStatusLabel.setText(_translate("MainWindow",
                                                      "<html><head/><body><p><span style=\" font-weight:600; "
                                                      "color:#00aa00;\">Status: Hartă încărcată</span></p></body>"
                                                      "</html>"))

            self.ui.locChooseComboBox.setEnabled(True)
            self.ui.locChooseLabel.setEnabled(True)
            self.ui.locChooseComboBox.clear()
            self.ui.locChooseComboBox.addItems(settlements_list)

            self.ui.droneWattPerKmLabel.setEnabled(True)
            self.ui.droneWattPerKmSpinBox.setEnabled(True)

            self.ui.locGenCourseButton.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    interfata = InterfataAplicatie()

    sys.exit(app.exec_())
