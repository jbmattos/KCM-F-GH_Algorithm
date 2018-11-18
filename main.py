import pandas as pd
from functions import *


def main():

    # INPUTS AND INITIAL SETTINGS
    data = pd.read_csv('datasets/segmentation.test.txt', delimiter=',')
    view_classes = data['CLASS']

    n_view = 2                      # 1:complete view   2:shape_view    3:rgb_view
    view = get_view(data, n_view)

    c = 7                           # number of clusters
    n = len(view)                   # number of examples
    p = len(view.iloc[0])           # number of attributes (dimensionality)

    # sigma2 = get_sigma2(view.values)    # sigma2_view2 = 126.56
    sigma2 = 126.56
    gama = (1/sigma2)**p


    # INITIALIZATION
    clusters = cluster_initialization(c, n)  # vector of cluster objects
    hp_vector = np.full(p, fill_value=gama**(1/p))      # hyper-parameter vector

    kernel_matrix, clusters = generate_kernel_matrix(view.values, hp_vector, clusters)
    distances_matrix = generate_distances(kernel_matrix, clusters)

    clusters, examples_location = generate_partition(kernel_matrix, distances_matrix, clusters)
    objective_function = get_objective_fnc(clusters, distances_matrix)
    print('Objective function = ', objective_function)


    # PARTITION ITERATIONS
    test = 1
    while test == 1:

        kernel_matrix, clusters = generate_kernel_matrix(view.values, hp_vector, clusters)
        hp_vector = hyper_parameter_updating(view.values, clusters, p, gama)

        distances_matrix = generate_distances(kernel_matrix, clusters)

        clusters, examples_new_location = generate_partition(kernel_matrix, distances_matrix, clusters)
        objective_function = get_objective_fnc(clusters, distances_matrix)
        print('Objective function = ', objective_function)

        print('Examples location equal comparison: ', np.array_equal(examples_location, examples_new_location))
        if np.array_equal(examples_location, examples_new_location):
            test = 0

        examples_location = examples_new_location[:]


if __name__ == '__main__':
    main()
