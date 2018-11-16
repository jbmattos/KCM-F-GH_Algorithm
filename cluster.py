import random
import numpy as np


class Cluster:

    def __init__(self, no_of_examples):
        self.prototype = []
        self.size = None
        self.kernel_matrix = None
        self.kernel = 0

        self.cluster_init(no_of_examples)

    def cluster_init(self, n):
        self.prototype.append(random.randint(0, n))
        self.size = len(self.prototype)

        return

    def set_cluster_kernel(self, kernel_matrix):

        n = self.size
        cluster_kernel_matrix = np.zeros(shape=(n, n))

        for e1_idx, element_1 in enumerate(self.prototype[:-1]):
            for e2_idx, element_2 in enumerate(self.prototype[(e1_idx+1):]):
                kernel = kernel_matrix[element_1, element_2]
                cluster_kernel_matrix[e1_idx, e2_idx] = kernel  # triangular matrix: superior
                cluster_kernel_matrix[e2_idx, e1_idx] = kernel  # triangular matrix: inferior

        self.kernel_matrix = cluster_kernel_matrix[:]
        self.kernel = np.sum(cluster_kernel_matrix)

        return