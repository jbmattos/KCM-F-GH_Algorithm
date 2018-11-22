import random
import json
import numpy as np
from cluster import Cluster
from sklearn.metrics.pairwise import euclidean_distances


def get_view(data, n_view):                                     # input: (pd.dataframe, int)

    if n_view == 1:
        drop_index = [0, 3, 4, 5]
        view = data.drop(data.columns[drop_index], axis=1)
        return view

    if n_view == 2:
        drop_index = [0, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        view = data.drop(data.columns[drop_index], axis=1)
        return view

    if n_view == 3:
        drop_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        view = data.drop(data.columns[drop_index], axis=1)
        return view

    return


def get_sigma2(data):                                           # input: (pd.dataframe.values)

    distances = euclidean_distances(data, data)
    n = len(distances)

    distances_vector = []
    i = 1
    for line in range(n):
        values = distances[line, i:]
        distances_vector = distances_vector + list(values)
        i += 1

    percents = [10, 90]
    percentiles = np.percentile(distances_vector, percents)

    return np.mean(percentiles)


def calculate_kernel(ex_1, ex_2, hyper_parameter):              # input: (np.array, np.array, np.array)

    kernel = np.sum(np.multiply(np.power((ex_1 - ex_2), 2), hyper_parameter))

    return np.exp(-0.5 * kernel)


def generate_kernel_matrix(data, hyper_parameter, clusters):              # input: (pd.dataframe.values, np.array)

    n = len(data)
    kernel_matrix = np.zeros(shape=(n, n))

    for ex_1 in range(n):
        for ex_2 in range(ex_1, n):
            kernel = calculate_kernel(data[ex_1], data[ex_2], hyper_parameter)
            kernel_matrix[ex_1, ex_2] = kernel  # triangular matrix: superior
            kernel_matrix[ex_2, ex_1] = kernel  # triangular matrix: inferior

    for cluster in clusters:
        cluster.set_cluster_kernel(kernel_matrix)

    return kernel_matrix, clusters


def cluster_initialization(no_of_clusters, no_of_examples):     # input: (int, int)

    clusters = []
    clusters_initial_prototypes = random.sample(range(no_of_examples), no_of_clusters)

    for c in range(no_of_clusters):
        prototype = clusters_initial_prototypes[c]
        cluster_obj = Cluster(prototype)
        clusters.append(cluster_obj)

    return clusters


def generate_distances(kernel_matrix, clusters):                # input: (np.matrix, list_of_objects)

    n = len(kernel_matrix)
    c = len(clusters)

    distances_matrix = np.zeros(shape=(n, c))

    for example in range(n):                            # iterating all examples

        for c_idx, cluster in enumerate(clusters):      # iterating all clusters (for each example)
            kernel_sum = 0
            for element in cluster.prototype:
                kernel_sum += kernel_matrix[example, element]
            distances_matrix[example, c_idx] = 1 - (2 * (kernel_sum/cluster.size)) + cluster.kernel/(cluster.size ** 2)
            if distances_matrix[example, c_idx] < 0:
                print('Menor que 0')

    return distances_matrix


def generate_partition(kernel_matrix, distances_matrix, clusters):  # input: (np.matrix, np.matrix, list_of_objects)

    elements_cluster_location = distances_matrix.argmin(axis=1)
    n = len(elements_cluster_location)

    for c_idx, cluster in enumerate(clusters):

        cluster_elements = []
        elements = np.argwhere(elements_cluster_location == c_idx)

        for e in elements:
            cluster_elements.append(e[0])

        if not cluster_elements:
            cluster_new_ex = random.randint(0, n - 1)
            old_cluster_ex = elements_cluster_location[cluster_new_ex]
            elements_cluster_location[cluster_new_ex] = c_idx
            if old_cluster_ex < c_idx:
                clusters[old_cluster_ex].prototype.remove(cluster_new_ex)

        cluster.prototype = cluster_elements[:]
        cluster.size = len(cluster_elements)

    return clusters, elements_cluster_location


def hyper_parameter_updating(data, clusters, p, gama):      # input: (pd.dataframe.values, list_of_objects, int, float)

    main_vector = np.zeros(shape=(1, p))
    for cluster in clusters:

        sum_vector = np.zeros(shape=(1, p))

        for e1_idx, element_1 in enumerate(cluster.prototype):
            for e2_idx, element_2 in enumerate(cluster.prototype):
                sum_vector += cluster.kernel_matrix[e1_idx, e2_idx] * ((data[element_1] - data[element_2]) ** 2)

        main_vector += sum_vector / cluster.size

    s_vector = np.zeros(shape=(1, p))
    for j in range(p):
        if main_vector[0, j] == 0:
            print('Division for Zero')
        s_vector[0, j] = (gama ** (1/p)) * (np.prod(main_vector) ** (1/p)) / main_vector[0, j]

    return s_vector


def get_objective_fnc(clusters, distances_matrix):

    objective_fnc = 0

    for c_idx, cluster in enumerate(clusters):

        dist_sum = 0
        for element in cluster.prototype:
            dist_sum += distances_matrix[element, c_idx]

        objective_fnc += dist_sum

    return objective_fnc


def print_results(partition, clusters, hp_vector, rand_idx, objective_function):
    print('\n\nBEST PARTITION: ', partition)
    print('Objective function ', objective_function)
    print('Adjusted rand index = ', rand_idx)
    print('Hyper-parameter vector: ', hp_vector)

    for c_idx, cluster in enumerate(clusters):
        print('\nCluster ', c_idx, ':')
        print('Number of objects: ', cluster.size)
        print('List of objects:\n', cluster.prototype)

    return


def save_results(file, partition, clusters, hp_vector, rand_idx, objective_function, examples_location):

    f = open(file, "a+")
    f.write('\n\n>> PARTITION: ' + repr(partition))
    f.write('\n\nObjective function = ' + repr(objective_function))
    f.write('\nAdjusted rand index = ' + repr(rand_idx))
    f.write('\nHyper-parameter vector: ' + repr(hp_vector))
    f.write('\nExamples location: ' + repr(list(examples_location)))
    for c_idx, cluster in enumerate(clusters):
        f.write('\n\nCluster ' + repr(c_idx) + ':')
        f.write('\nNumber of objects: ' + repr(cluster.size))
        f.write('\nList of objects:\n' + repr(cluster.prototype))
    f.close()

    # with open('result.json', 'w+') as fp:   # saves last elements-cluster position
    #     json.dump(examples_location, fp)

    return


def z_score_normalization(view):        # input: (pd.dataframe)

    p = len(view.iloc[0])
    # data = view.values
    mean = np.mean(view.values, axis=0)
    std = np.std(view.values, axis=0)

    for col in range(p):
        attr = view.values[:, col]
        norm_data = np.subtract(attr, mean[col]) / std[col]
        view.values[:, col] = norm_data[:]

    return view
