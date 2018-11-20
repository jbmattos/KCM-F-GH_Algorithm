import pandas as pd
from functions import *
from sklearn.metrics import adjusted_rand_score


def main():

    # INPUTS AND INITIAL SETTINGS
    data = pd.read_csv('datasets/segmentation.test.txt', delimiter=',')
    view_classes = data['CLASS']

    n_view = 2                      # 1:complete view   2:shape_view    3:rgb_view
    view = get_view(data, n_view)

    c = 7                           # number of clusters
    n = len(view)                   # number of examples
    p = len(view.iloc[0])           # number of attributes (dimensionality)

    sigma2 = get_sigma2(view.values)    # sigma2_view2 = 126.56
    gama = (1/sigma2)**p


    # INITIALIZATION
    clusters = cluster_initialization(c, n)  # vector of cluster objects
    hp_vector = np.full(p, fill_value=gama**(1/p))      # hyper-parameter vector

    kernel_matrix, clusters = generate_kernel_matrix(view.values, hp_vector, clusters)
    distances_matrix = generate_distances(kernel_matrix, clusters)

    clusters, examples_location = generate_partition(kernel_matrix, distances_matrix, clusters)
    objective_function = get_objective_fnc(clusters, distances_matrix)
    print('Iteration: 0')
    print('Objective function = ', objective_function, '\n')


    # PARTITION ITERATIONS
    test = 1
    it = 1
    while test == 1:

        print('Iteration: ', it)

        kernel_matrix, clusters = generate_kernel_matrix(view.values, hp_vector, clusters)
        hp_vector = hyper_parameter_updating(view.values, clusters, p, gama)

        distances_matrix = generate_distances(kernel_matrix, clusters)

        clusters, examples_new_location = generate_partition(kernel_matrix, distances_matrix, clusters)
        objective_function = get_objective_fnc(clusters, distances_matrix)
        print('Objective function = ', objective_function, '\n')

        if np.array_equal(examples_location, examples_new_location):
            print('Clusters converged: end while')
            test = 0

        examples_location = examples_new_location[:]
        it += 1

    rand_idx = adjusted_rand_score(view_classes, examples_location)
    print('Rand index = ', rand_idx)


if __name__ == '__main__':
    main()
