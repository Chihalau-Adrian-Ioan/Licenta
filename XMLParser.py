import concurrent.futures
import math
import queue
import random
import threading
import time
import xml.etree.ElementTree as ET
from itertools import repeat

import networkx as nx
import numpy as np


# Folosind algoritmul de căutare în adâncime, caută insulele din graf (subgrafuri care nu sunt conectate intre ele)
# si returnează nodurile care nu fac parte din insula cu cele mai multe noduri, adică nodurile izolate
# metodă bazată pe rezolvarea din https://en.wikipedia.org/wiki/Breadth-first_search#:~:text=procedure%20BFS(,enqueue(w)
def find_isolated_nodes(nodes_dict: dict, edges_dict: dict):
    nodes_count = len(list(nodes_dict.keys()))
    marked_island_nodes = np.full(nodes_count, 0)  # vectorul de apartenență a nodurilor la o insulă,
    # prin marcarea cu numărul de ordine specific insulei

    island_index = 0  # indexul curent al insulei
    busiest_island_index = None  # indexul insulei cu cele mai multe noduri
    busiest_island_count = 0  # numarul de noduri din subgraful cu cele mai multe din ele

    # cat timp inca exista noduri care nu au fost vizitate de o insula
    while np.any(marked_island_nodes == 0):
        # se găsește primul nod care nu a fost vizitat inca
        island_start_node = None
        for node in range(len(marked_island_nodes)):
            if marked_island_nodes[node] == 0:
                island_start_node = node
                break

        # si este considerat ca punctul de plecare dintr-o noua insula
        if island_start_node is not None:
            island_index += 1
            current_island_count = 1
            marked_island_nodes[island_start_node] = island_index
            island_queue = queue.Queue()
            island_queue.put(island_start_node)

            # se parcurge in adancime insula, pornind de la nodul de plecare
            while not island_queue.empty():
                current_node = island_queue.get()
                for edge_d in edges_dict.keys():
                    node_a, node_b = edge_d
                    # daca se gaseste un nod adiacent cu nodul curent,
                    # atunci acesta face parte din insula cu indexul curent
                    if node_a == current_node and marked_island_nodes[node_b] == 0:
                        marked_island_nodes[node_b] = island_index
                        island_queue.put(node_b)
                        current_island_count += 1
                    elif node_b == current_node and marked_island_nodes[node_a] == 0:
                        marked_island_nodes[node_a] = island_index
                        island_queue.put(node_a)
                        current_island_count += 1

            # daca dupa vizitarea nodurilor din insula curenta, s-a gasit un numar mai mare
            # decat insula curenta cu cele mai multe noduri,
            # atunci devine noua insula aglomerata
            if current_island_count > busiest_island_count:
                busiest_island_count = current_island_count
                busiest_island_index = island_index

    # se returneaza nodurile izolate, adica cele care nu fac parte din insula cu cele mai multe noduri
    return [node for node in range(nodes_count) if marked_island_nodes[node] != busiest_island_index]


# algoritmul Hierholzer de determinare a ciclului eulerian dintr-un graf eulerian
def hierholzer_alg(nodes_dict: dict, edges_dict: dict):
    # copiem dictionarele de noduri, respectiv muchii (pentru a nu denatura originalele)
    nodes_dict_copy = {key: val for key, val in nodes_dict.items()}
    edges_dict_copy = {key: val for key, val in edges_dict.items()}

    start_node = random.choice(list(nodes_dict.keys()))  # punctul de start al turului, ales aleatoriu
    total_edges = sum(edges_dict[edge_d]['freq'] for edge_d in edges_dict.keys())  # numarul total de muchii din graf

    tour = []  # turul propriu-zis
    visited_edges = 0  # numar de muchii vizitate

    while visited_edges < total_edges:
        # se determina nodul curent de start
        current_start_node = None
        # la inceput, turul principal va avea ca prim element nodul de start ales
        if len(tour) == 0:
            current_start_node = start_node
        # daca un tur a fost deja realizat, dar nu au fost vizitate toate muchiile
        else:
            # se gaseste primul nod din tur care are gradul nenul (inca mai are muchii adiacente nevizitate)
            nodes_g = set(tour)
            for node_n in nodes_g:
                if nodes_dict_copy[node_n]['grade'] > 0:
                    current_start_node = node_n
                    break

        # odata ce a fost gasit un nod de start pentru un tur
        if current_start_node is not None:
            # este creat un nou tur
            new_tour = [current_start_node]

            chosen_edge = None
            lowest_freq = np.Inf
            # si se gaseste prima muchie adiacenta cu nodul de start, cu frecventa cea mai mica
            for g_edge in edges_dict_copy.keys():
                node_n, node_m = g_edge
                # cand este gasit o muchie cu frecventa pozitiva (inca mai poate fi vizitata)
                # si mai mica decat minimul curent
                if (node_n == new_tour[-1] or node_m == new_tour[-1]) \
                        and 0 < edges_dict_copy[g_edge]['freq'] < lowest_freq:
                    # se pastreaza muchia si se inlocuieste minimul cu noua frecventa
                    lowest_freq = edges_dict_copy[g_edge]['freq']
                    chosen_edge = g_edge

            # La finalul buclei for se va gasi o muchie cu frecventa cea mai mica
            # Astfel ne asiguram sa trecem prin muchiile cu frecventa 1 mai intai, pentru a nu fi vizitate din nou in
            # alte tururi
            # Se adauga muchia cu cea mai mica frecventa strict pozitiva la turul curent
            node_n, node_m = chosen_edge
            new_tour.append(node_n if node_n != new_tour[-1] else node_m)
            visited_edges += 1
            # se scad de asemenea gradele capetelor muchiei, cat si frecventa acesteia
            nodes_dict_copy[node_n]['grade'] -= 1
            nodes_dict_copy[node_m]['grade'] -= 1
            edges_dict_copy[chosen_edge]['freq'] -= 1

            # cat timp ultimul nod din tur nu coincide cu nodul de start
            while new_tour[-1] != current_start_node:
                chosen_edge = None
                lowest_freq = np.Inf
                # se repeta procedura de gasire a unei muchii, ca cea de mai sus,
                # dar de data asta cu o muchie adiacenta cu ultimul nod din tur
                for g_edge in edges_dict_copy.keys():
                    node_n, node_m = g_edge
                    if (node_n == new_tour[-1] or node_m == new_tour[-1]) \
                            and 0 < edges_dict_copy[g_edge]['freq'] < lowest_freq:
                        lowest_freq = edges_dict_copy[g_edge]['freq']
                        chosen_edge = g_edge

                node_n, node_m = chosen_edge
                nodes_dict_copy[node_n]['grade'] -= 1
                nodes_dict_copy[node_m]['grade'] -= 1
                edges_dict_copy[chosen_edge]['freq'] -= 1
                new_tour.append(node_n if node_n != new_tour[-1] else node_m)
                visited_edges += 1

            # daca este prima oara cand se realizeaza un tur
            if len(tour) == 0:
                tour = new_tour.copy()
            else:
                # altfel se gaseste pozitia nodului curent de start in tur
                start_pos_index = tour.index(current_start_node)
                # iar apoi se divide turul in 2, excluzand pozitia nodului curent de start
                if start_pos_index == 0:
                    tour_half1, tour_half2 = [], tour[start_pos_index + 1:].copy()
                elif start_pos_index == len(tour) - 1:
                    tour_half1, tour_half2 = tour[:start_pos_index].copy(), []
                else:
                    tour_half1, tour_half2 = tour[:start_pos_index].copy(), tour[start_pos_index + 1:].copy()

                # in cele din urma, se reconstruieste turul initial, adaugand noul tur in locul pozitiei gasite
                tour = tour_half1.copy()
                tour.extend(new_tour)
                tour.extend(tour_half2)

    # odata ce s-au vizitat toate muchiile, se returneaza rezultatul
    return tour


# functie de determinare a nodurilor din drumul dintre 2 noduri, folosind matricea obtinuta in algoritmul Floyd-Warshall
# algoritm preluat din
# https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm#:~:text=procedure%20Path(u%2C%20v)%0A%20%20%20%20if%20next%5Bu%5D%5Bv%5D%20%3D%20null%20then%0A%20%20%20%20%20%20%20%20return%20%5B%5D%0A%20%20%20%20path%20%E2%86%90%20%5Bu%5D%0A%20%20%20%20while%20u%20%E2%89%A0%20v%0A%20%20%20%20%20%20%20%20u%20%E2%86%90%20next%5Bu%5D%5Bv%5D%0A%20%20%20%20%20%20%20%20path.append(u)%0A%20%20%20%20return%20path
def floyd_warshall_path(u: int, v: int, odd_nodes_dict):
    if odd_nodes_dict[u]['next_nodes'][v] is None:
        return []
    the_path = [u]
    alt_u = u
    while alt_u != v:
        alt_u = odd_nodes_dict[alt_u]['next_nodes'][v]
        the_path.append(alt_u)
    return the_path


# algoritmul Floyd-Warshall de determinare a tuturor drumurilor minime
# returneaza un vector de etichetare a fiecarui nod din graf, matricea de distante minime intre fiecare nod,
# cat si matricea de adiacenta, pentru a reconstrui drumul realizat
# algoritm bazat pe link-ul
# https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm#:~:text=%5Bedit%5D-,let%20dist%20be%20a,i%5D%5Bk%5D%20%2B%20dist%5Bk%5D%5Bj%5D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20next%5Bi%5D%5Bj%5D%20%E2%86%90%20next%5Bi%5D%5Bk%5D,-procedure%20Path(u
def floyd_warshall_alg(nodes_dict: dict, graph_edges_dict: dict):
    nodes_count = len(nodes_dict.keys())
    nodes_list = [graph_node for graph_node in nodes_dict.keys()]
    dists = np.full((nodes_count, nodes_count), np.inf)  # matricea de distante minime;
    # initial, toate au valoarea infinit

    next_node_m = np.full((nodes_count, nodes_count), None)  # matricea de adiacenta a nodurilor;
    # initial, toate au valorea None(adica nu sunt adiacente cu vreun nod in drum)

    # initializarea matricilor de distante minime, respectiv de adiacenta
    for u in range(nodes_count):
        for v in range(nodes_count):
            edge_g = (nodes_list[u], nodes_list[v])
            if u != v and edge_g in graph_edges_dict.keys():
                dists[u, v] = graph_edges_dict[edge_g]
                dists[v, u] = graph_edges_dict[edge_g]
                next_node_m[u, v] = v
                next_node_m[v, u] = u
            elif u == v:
                dists[u, v] = 0
                next_node_m[u, v] = v

    # determinarea propriu-zisa a distantelor minime, respectiv a adiacentei in drum
    for k in range(nodes_count):
        for i in range(nodes_count):
            for j in range(nodes_count):
                if dists[i, j] > dists[i, k] + dists[k, j]:
                    dists[i, j] = dists[i, k] + dists[k, j]
                    next_node_m[i, j] = next_node_m[i, k]

    return nodes_list, dists, next_node_m


def dijkstra_alg(nodes_dict: dict, edges_dict: dict, start_node):
    nodes_count = len(nodes_dict.keys())

    lowest_dists = np.full(nodes_count, np.Inf)  # lista celor mai mici distanțe între nodul de plecare(start_node)
    # și celelalte noduri din graf; inițial, acestea sunt distanțe infinite (maxime)

    prev_node = np.full(nodes_count, None)  # lista noduri precedente drumului celui mai scurt între nodul start_node și
    # celelalte noduri din graf; inițial nu se cunosc drumurile (deci nu au precedent)

    visited_nodes = np.full(nodes_count, False)  # lista de booleane, visited_node[i] reprezentând dacă nodul de indexul
    # i din nodes_list a fost vizitat sau nu în cadrul algoritmului lui Dijkstra

    lowest_dists[start_node] = 0

    while not np.all(visited_nodes):
        min_node_index = min([(index, lowest_dists[index]) for index in range(nodes_count)
                              if visited_nodes[index] == False], key=lambda x: x[1])[0]
        visited_nodes[min_node_index] = True

        for node_index in range(nodes_count):
            if not visited_nodes[node_index]:
                edge = None
                if (node_index, min_node_index) in edges_dict.keys():
                    edge = (node_index, min_node_index)
                elif (min_node_index, node_index) in edges_dict.keys():
                    edge = (min_node_index, node_index)
                if edge is not None:
                    dist = lowest_dists[min_node_index] + edges_dict[edge]
                    if dist < lowest_dists[node_index]:
                        lowest_dists[node_index] = dist
                        prev_node[node_index] = min_node_index

    return lowest_dists, prev_node


# algoritmul de construire al unui drum, pornind de la nodul de origine și ajungând la alt nod
# în dicționarul de noduri este stocat pentru fiecare nod vectorul de precdenți în drumurile ce pornesc din el
def dijkstra_path_odd(u, v, nodes_dict) -> list:
    if u == v:
        return [u]
    else:
        rec_lst = dijkstra_path_odd(u, nodes_dict[u]['prev_nodes'][v], nodes_dict)
        rec_lst.append(v)
        return rec_lst


def get_city_graph_from_map(filename, city_name):
    # prelucrarea informatiilor hartii pentru a prelua graful acestuia
    tree = ET.parse(filename)
    root = tree.getroot()
    node_dict = {}

    # extragerea nodurilor din graf
    # Nota: nu toate nodurile reprezinta intersectii, unele pot fi chiar noduri ce determina
    # colturile unui parc sau a unei cladiri
    for node in root.iter('node'):
        node_dict[node.attrib['id']] = {'lat': float(node.attrib['lat']), 'lon': float(node.attrib['lon'])}

    street_nodes_dict = {}
    streets_dict = {}
    graph_edges_dict = {}

    # extragerea strazilor din graf
    # in formatul osm, strazilor fac parte din tag-urile way (căi), care pot reprezenta si perimetrele unor cladiri
    for child in root.findall('way'):
        highway, name, in_city, nodes = None, (), None, []

        # De aceea, se cauta căile care sunt străzi principale (drumuri nationale, europene,autostrazi),
        # secundare, tertiare si rezidentiale (strazi din oras), care sunt catalogate aparținând orașului dorit
        for tag in child.iter('tag'):
            if tag.attrib['k'] == 'highway' and tag.attrib['v'] in ['primary', 'secondary', 'tertiary', 'residential']:
                highway = tag.attrib['v']
            elif tag.attrib['k'] == 'is_in:city':
                in_city = tag.attrib['v']
            elif tag.attrib['k'] == 'name':
                name = tag.attrib['v']

        # daca a fost gasita o astfel de strada
        if in_city == city_name and highway is not None:
            # toate nodurile din acesta sunt catalogate drept noduri (intersectii) ce apartin grafului orasului
            for node in child.iter('nd'):
                node_id = node.attrib['ref']
                if node_id not in street_nodes_dict.keys():
                    street_nodes_dict[node_id] = node_dict[node_id].copy()
                    street_nodes_dict[node_id]['grade'] = 0
                nodes.append(node.attrib['ref'])
            if child.attrib['id'] not in streets_dict.keys():
                streets_dict[child.attrib['id']] = {'name': name, 'highway': highway, 'nodes': nodes}

    # totusi, trebuie determinate distantele intre fiecare intersectie din oras
    # se foloseste formula haversina, ce calculeaza distanta intre 2 puncte aflate pe glob
    # formula preluată din http://www.movable-type.co.uk/scripts/latlong.html
    R = 6371  # raza Pământului, în kilometrii
    for street in streets_dict.keys():
        nodes = streets_dict[street]['nodes']
        for node1, node2 in zip(nodes[:-1], nodes[1:]):
            lat1, lon1 = street_nodes_dict[node1]['lat'], street_nodes_dict[node1]['lon']
            lat2, lon2 = street_nodes_dict[node2]['lat'], street_nodes_dict[node2]['lon']

            # se convertesc coordonatele in radiani
            phi1, phi2 = lat1 * math.pi / 180, lat2 * math.pi / 180
            lambda1, lambda2 = lon1 * math.pi / 180, lon2 * math.pi / 180

            diff_lambda = lambda2 - lambda1
            diff_phi = phi2 - phi1

            # se foloseste formula haversina, ce calculeaza distanta intre 2 puncte aflate pe glob
            # a = patratul jumatatii din lungimea arcului cercului dintre 2 puncte
            # c = distanta unghiulara in radiani
            # d = distanta obtinuta
            a = math.sin(diff_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(diff_lambda / 2) ** 2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            d = R * c

            # pe langa adaugarea muchiei in graf, se maresc si gradele celor 2 noduri
            graph_edges_dict[(node1, node2)] = d
            street_nodes_dict[node1]['grade'] += 1
            street_nodes_dict[node2]['grade'] += 1

    # se simplifică notațiile nodurilor și muchiilor incidente prin asignarea fiecăruia cu câte un număr de ordine
    node_index_dict = {graph_node: index for index, graph_node in enumerate(list(street_nodes_dict.keys()))}
    indexed_nodes_dict = {node_index_dict[key]: value for key, value in street_nodes_dict.items()}
    indexed_edges_dict = {(node_index_dict[edge[0]], node_index_dict[edge[1]]): value
                          for edge, value in graph_edges_dict.items()}

    return indexed_nodes_dict, indexed_edges_dict


def greedy_eulerian_graph(nodes_dict, edges_dict, odd_pairs_distances, odd_nodes_dict, fd_or_djk):
    # metoda Greedy: se extrag cele mai mici distante intre fiecare pereche de noduri de grad impar, tinand cont ca
    # marginile sa nu se repete (numar de muchii = numar de noduri grad impar / 2)
    distances = [(dist, val) for dist, val in odd_pairs_distances.items()]
    distances.sort(key=lambda x: x[1])

    selected_nodes = []
    selected_distances = []
    for dist, val in distances:
        node1, node2 = dist
        if node1 not in selected_nodes and node2 not in selected_nodes:
            selected_distances.append((dist, val))
            selected_nodes.append(node1)
            selected_nodes.append(node2)

    # crearea noului graf de strazi, adaugand muchiile determinate in Procedura Greedy

    # initial, nodurile au acelasi grad
    # iar muchiile vor avea in plus o valoare de frecventa
    # (de cate ori se poate parcurge nodul in circuitul eulerian)
    for distance in edges_dict.keys():
        edges_dict[distance] = {'d': edges_dict[distance], 'freq': 1}

    for dist, val in selected_distances:
        node1, node2 = dist
        # se obtine calea drumului minim dintre nodurile alese
        node_path = dijkstra_path_odd(node1, node2, odd_nodes_dict) if fd_or_djk is False \
            else floyd_warshall_path(node1, node2, nodes_dict)
        for l_node, r_node in zip(node_path[:-1], node_path[1:]):
            # daca muchia dintre cele 2 noduri adiacente exista deja
            dist_changed = None
            if (l_node, r_node) in edges_dict.keys():
                dist_changed = (l_node, r_node)
            elif (r_node, l_node) in edges_dict.keys():
                dist_changed = (r_node, l_node)

            if dist_changed is not None:
                # se mareste frecventa acesteia, si cresc gradele capetelor
                edges_dict[dist_changed]['freq'] += 1
                nodes_dict[l_node]['grade'] += 1
                nodes_dict[r_node]['grade'] += 1


def generate_course(source_filename, settlement_name, roadExtractionEvent: threading.Event,
                    routeProcessingEvent: threading.Event,
                    animationProcessingEvent: threading.Event, stopEvent: threading.Event, solution_content: dict):
    roadExtractionEvent.set()
    # se extrage din harta graful orasului
    nodes_dict, edges_dict = get_city_graph_from_map(source_filename, settlement_name)

    # se cauta noduri care sunt deconectate de la graf, folosind BFS
    isolated_nodes = find_isolated_nodes(nodes_dict, edges_dict)

    # daca au fost gasite noduri izolate
    if len(isolated_nodes) > 0:
        print(f'Isolated nodes: {isolated_nodes}')
        marked_edges = []
        for edge in edges_dict:
            node1, node2 = edge
            if node1 in isolated_nodes or node2 in isolated_nodes:
                marked_edges.append(edge)

        for edge in marked_edges:
            node1, node2 = edge
            edges_dict.pop(edge)
            nodes_dict[node1]['grade'] -= 1
            nodes_dict[node2]['grade'] -= 1

        for isolated_node in isolated_nodes:
            nodes_dict.pop(isolated_node)

        # calibrare dicționare de noduri și muchii după eliminarea nodurilor izolate și muchiilor incidente
        node_index_dict = {graph_node: index for index, graph_node in enumerate(list(nodes_dict.keys()))}
        nodes_dict = {node_index_dict[key]: value for key, value in nodes_dict.items()}
        edges_dict = {(node_index_dict[edge[0]], node_index_dict[edge[1]]): value
                      for edge, value in edges_dict.items()}
    else:
        print('No isolated nodes found')

    routeProcessingEvent.set()
    # se verifica daca graful este eulerian
    # daca toate nodurile din graf au gradul par
    is_eulerian = True

    for node in nodes_dict.keys():
        if nodes_dict[node]['grade'] % 2 == 1:
            is_eulerian = False
            break

    print(f'Is Eulerian? Response: {is_eulerian}')

    # in cazul in care graful nu este eulerian, avem de a face cu problema postasului chinez
    # (determinarea unui cost suplimentar minim al muchiilor astfel incat graful sa devina eulerian)
    if not is_eulerian:
        # se determina distanta minima a drumurilor intre fiecare pereche posibilă de noduri de grad impar din graful
        # orasului, cat si predecesorii fiecărui nod, pentru reconstrui drumurile
        # se determina nodurile de grad impar, adaugându-le într-un dicționar (se vor pastra aici si vectorii de
        # costuri minime, respectiv de precedență a drumurilor între celelalte noduri)
        odd_nodes_dict = {}
        for node_index in nodes_dict.keys():
            if nodes_dict[node_index]['grade'] % 2 == 1:
                odd_nodes_dict[node_index] = {'info': nodes_dict[node_index]}

        start_time = time.time()

        fd_or_djk = False  # daca se aplica floyd-warshall sau dijkstra

        if fd_or_djk:
            # se extrag matricile de distante minime, respectiv de succesori din algoritmul floyd-warshall
            new_nodes_list, dist_nodes, next_node_matrix = floyd_warshall_alg(nodes_dict, edges_dict)

            # recalibrare dictionare in functie de notatia noua din new_nodes_dict
            node_index_dict = {graph_node: index for index, graph_node in enumerate(new_nodes_list)}
            nodes_dict = {node_index_dict[key]: value for key, value in nodes_dict.items()}
            edges_dict = {(node_index_dict[edge[0]], node_index_dict[edge[1]]): value
                          for edge, value in edges_dict.items()}

            # introducerea vectorilor de drumuri minime, respectiv succesori pentru fiecare nod, în acestia
            for node_i in nodes_dict.keys():
                nodes_dict[node_i]['dist_nodes'] = dist_nodes[node_i]
                nodes_dict[node_i]['next_nodes'] = next_node_matrix[node_i]

                if node_i in odd_nodes_dict.keys():
                    odd_nodes_dict[node_i]['dist_nodes'] = dist_nodes[node_i]
        else:
            # se prelucrează în paralel, folosind un executor de procese, vectorii de distante, cât și de precedenți,
            # pentru fiecare nod de grad impar
            with concurrent.futures.ProcessPoolExecutor() as executor:
                nodes_list = list(odd_nodes_dict.keys())
                for node_i, result in zip(nodes_list, executor.map(dijkstra_alg,
                                                                   repeat(nodes_dict), repeat(edges_dict),
                                                                   nodes_list)):
                    dist_nodes, prev_nodes = result
                    odd_nodes_dict[node_i]['dist_nodes'] = dist_nodes
                    odd_nodes_dict[node_i]['prev_nodes'] = prev_nodes
        end_time = time.time()

        print(f'Execution: {(end_time - start_time)} seconds')
        print(f'Number of nodes: {len(nodes_dict.keys())}')
        print(f'Number of edges: {len(edges_dict.keys())}')

        # se determina dictionarul de distante intre fiecare pereche posibila de noduri de grad impar
        odd_pairs_distances = {}

        for odd_node1 in odd_nodes_dict.keys():
            for odd_node2 in odd_nodes_dict.keys():
                if odd_node1 != odd_node2 and (odd_node2, odd_node1) not in odd_pairs_distances.keys() \
                        and odd_nodes_dict[odd_node1]['dist_nodes'][odd_node2] != np.Inf:
                    odd_pairs_distances[(odd_node1, odd_node2)] = odd_nodes_dict[odd_node1]['dist_nodes'][odd_node2]

        print('\n')

        # Folosind procedura greedy, se modifică graful orașului astfel încât acesta să devină eulerian
        greedy_eulerian_graph(nodes_dict, edges_dict, odd_pairs_distances, odd_nodes_dict, fd_or_djk)

        # la final se verifica inca o data daca este eulerian, pentru a demonstra corectitudinea solutiei
        print()
        isEulerian = True
        odd_nodes_count = 0
        for node in nodes_dict.keys():
            if nodes_dict[node]['grade'] % 2 != 0:
                isEulerian = False
                break
        print(f'Is Eulerian? {isEulerian}')
        print(f'Odd nodes count: {odd_nodes_count}')

    solution_nodes = nodes_dict.copy()
    solution_edges = edges_dict.copy()

    # determinarea circuitului eulerian
    solution = hierholzer_alg(solution_nodes, solution_edges)

    # pregătirea grafului si a solutiei pentru reprezentarea grafică în GUI
    animationProcessingEvent.set()
    # metodă preluată din http://brooksandrew.github.io/simpleblog/articles/intro-to-graph-optimization-solving-cpp/
    # creare graf gol
    g = nx.Graph()

    # adaugarea muchiilor(strazilor) grafului specific orasului
    for edge, val in solution_edges.items():
        node1, node2 = edge
        g.add_edge(node1, node2, d=val['d'])

    for node, val in solution_nodes.items():
        nx.set_node_attributes(g, {node: {'lat': val['lat'], 'lon': val['lon']}})

    print(f'Eulerian circuit: {solution}')

    solution_content['graph'] = g
    solution_content['solution'] = solution
    solution_content['lungime_totala_strazi'] = 0
    solution_content['distanta_parcursa_suplimentar'] = 0
    solution_content['distanta_totala_tur'] = 0
    solution_content['distanta parcursa drona curenta'] = [0]

    for strada in solution_edges.keys():
        solution_content['lungime_totala_strazi'] += solution_edges[strada]['d']

    solution_content['lungime_totala_strazi'] = round(solution_content['lungime_totala_strazi'], 3)

    total_sol_length = 0
    for node1, node2 in zip(solution[:-1], solution[1:]):
        if (node1, node2) in solution_edges.keys():
            edge = (node1, node2)
        else:
            edge = (node2, node1)

        solution_content['distanta parcursa drona curenta'].append(
            round(total_sol_length + solution_edges[edge]['d'], 3))
        total_sol_length += solution_edges[edge]['d']

    solution_content['distanta_totala_tur'] = round(total_sol_length, 3)
    solution_content['distanta_parcursa_suplimentar'] = round(solution_content['distanta_totala_tur'] -
                                                              solution_content['lungime_totala_strazi'], 3)

    stopEvent.set()
