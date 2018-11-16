import numpy as np
from cluster import Cluster
from sklearn.metrics.pairwise import euclidean_distances


def get_view(data, n_view):                                     # input: (pd.dataframe, int)

    if n_view == 1:
        view = data.drop(labels='CLASS', axis=1)
        return view

    if n_view == 2:
        drop_index = [0, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
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


def generate_kernel_matrix(data, hyper_parameter):              # input: (pd.dataframe.values, np.array)

    n = len(data)
    kernel_matrix = np.zeros(shape=(n, n))

    for ex_1 in range(n-1):
        for ex_2 in range(ex_1+1, n):
            kernel = calculate_kernel(data[ex_1], data[ex_2], hyper_parameter)
            kernel_matrix[ex_1, ex_2] = kernel  # triangular matrix: superior
            kernel_matrix[ex_2, ex_1] = kernel  # triangular matrix: inferior

    return kernel_matrix


def cluster_initialization(no_of_clusters, no_of_examples):     # input: (int, int)

    clusters = []
    for c in range(no_of_clusters):
        cluster_obj = Cluster(no_of_examples)
        clusters.append(cluster_obj)

    return clusters


def generate_distances(kernel_matrix, clusters):                # input: (np.matrix, list_of_objects)

    n = len(kernel_matrix)
    c = len(clusters)

    distances_matrix = np.zeros(shape=(n, c))

    for example in range(n):                            # iterating all examples

        kernel_sum = 0
        for c_idx, cluster in enumerate(clusters):      # iterating all clusters (for each example)
            for element in cluster.prototype:
                kernel_sum += kernel_matrix[example, element]
            distances_matrix[example, c_idx] = 1 - (2 * kernel_sum)/cluster.size + cluster.kernel/(cluster.size ** 2)

    return distances_matrix


def generate_partition(kernel_matrix, distances_matrix, clusters):  # input: (np.matrix, np.matrix, list_of_objects)

    elements_cluster_location = distances_matrix.argmin(axis=1)

    for c_idx, cluster in enumerate(clusters):

        cluster_elements = []
        elements = np.argwhere(elements_cluster_location == c_idx)

        for e in elements:
            cluster_elements.append(e[0])

        cluster.prototype = cluster_elements[:]
        cluster.size = len(cluster_elements)
        cluster.set_cluster_kernel(kernel_matrix)

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
        s_vector[0, j] = (gama ** (1/p)) * (np.prod(main_vector) ** (1/p)) / main_vector[0, j]

    return s_vector
