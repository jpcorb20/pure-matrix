from random import random
from math import sqrt
from typing import List


class Matrix:

    def __init__(self, lists: List[List[float]]):

        lengths = list(map(lambda x: len(x), lists))

        assert all([i == lengths[0] for i in lengths]), "All lists are not of same length."

        self._lists = lists
        self.shape = (len(lists), lengths[0])

    def __str__(self):
        return str(self.get_list())

    def __iter__(self):
        return iter(self._lists)

    def get_shape(self):
        return self.shape

    def get_list(self):
        return self._lists

    def get_element(self, i: int, j: int):

        assert i < self.shape[0] and j < self.shape[1], "Coordinates does not exist."

        return self._lists[i][j]

    def slice(self, i: tuple=None, j: tuple=None):

        if i is None:
            i = (0, self.shape[0])

        if j is None:
            j = (0, self.shape[1])

        assert len(i) == 2 and len(j) == 2 and min(i) >= 0 and min(j) >= 0 and \
            max(i) <= self.shape[0] and max(j) <= self.shape[1], "Coordinates does not exist."

        return Matrix([[self._lists[I][J] for J in list(range(*j))] for I in list(range(*i))])

    def get_flatten_list(self):
        return [j for i in self._lists for j in i]

    def transpose(self):
        return Matrix([[self._lists[i][j] for i in range(self.shape[0])] for j in range(self.shape[1])])

    def sum(self, axis: int=-1):

        assert axis in [-1, 0, 1], "Not a valid axis."

        if axis == 0:
            return Matrix([[sum(i)] for i in self._lists])
        elif axis == 1:
            return Matrix([[sum(i) for i in self.transpose()]])
        elif axis == -1:
            return sum([sum(i) for i in self._lists])

    def mean(self, axis: int=-1):

        assert axis in [-1, 0, 1], "Not a valid axis."

        if axis == 0 or axis == 1:
            return self.sum(axis=axis).scalar_product(1/float(self.shape[1-axis]))
        elif axis == -1:
            return self.sum(axis=-1) / (self.shape[0] * self.shape[1])

    def scalar_product(self, a: float):
        return Matrix([[a * j for j in i] for i in self._lists])

    def element_sum(self, a):

        assert self.shape == a.shape

        return Matrix([
            [j + q for j, q in zip(i, p)]
            for i, p in zip(self._lists, a.get_list())
        ])

    def element_substract(self, a):

        assert self.shape == a.shape

        return Matrix([
            [j - q for j, q in zip(i, p)]
            for i, p in zip(self._lists, a.get_list())
        ])

    def element_product(self, a):

        assert self.shape == a.shape

        return Matrix([
            [j * q for j, q in zip(i, p)]
            for i, p in zip(self._lists, a.get_list())
        ])

    def element_exponent(self, power: int=2):
        return Matrix([[j ** power for j in i] for i in self._lists])

    def element_divide(self, a):

        assert self.shape == a.shape

        assert all([j != 0 for i in a.get_list() for j in i]), "At least one element is zero."

        return Matrix([
            [j / q for j, q in zip(i, p)]
            for i, p in zip(self._lists, a.get_list())
        ])

    def column(self, j):
        assert j < self.shape[1], "Not enought columns"
        return Matrix([[self._lists[i][j]] for i in range(self.shape[0])])

    def row(self, i):
        assert i < self.shape[0], "Not enought rows"
        return Matrix([[self._lists[i][j] for j in range(self.shape[1])]])

    def mat_product(self, a):

        assert self.shape[1] == a.get_shape()[0]

        return Matrix([
            [self.row(i).element_product(a.column(j).transpose()).sum()
                for j in range(a.get_shape()[1])
            ] for i in range(self.shape[0])
        ])

    def is_square(self):
        return self.shape[0] == self.shape[1]

    def repeat_vector(self, n: int):
        s = self.get_shape()

        assert (s[0] != 1 or s[1] != 1), "One dimension must be 1 for vector."

        if s[0] == 1:
            return Matrix([self.get_list()[0] for _ in range(n)])
        elif s[1] == 1:
            return Matrix([[i[0] for _ in range(n)] for i in self.get_list()])

    def standardize(self, axis: int=1):
        assert axis in [0, 1], "Not a valid axis."
        return self.element_substract(self.mean(axis=axis).repeat_vector(n=self.shape[1 - axis]))

    def corr(self):
        centered = self.standardize()
        return centered.transpose().mat_product(centered)


class PCA:

    def __init__(self, n_components: int=2):
        self.n_components = n_components
        self.val, self.vecs = None, None

    def naive_power_iteration(self, A: Matrix, verbose: bool = False, tol: float = 1e-6, n_iter: int = 100):

        assert A.is_square(), "A is not square"

        n = A.get_shape()[0]

        b_v = list()
        b_vec = list()

        for _ in range(n):

            b_v_temp = None
            b_v_new = 0
            b = Matrix([[sqrt(1 / n)] for _ in range(n)])

            for i in range(n_iter):

                new_b = A.mat_product(b)
                b = new_b.element_divide(Matrix([[sqrt(new_b.element_exponent(2).sum())] for _ in range(n)]))

                b_v_new = b.transpose().mat_product(A.mat_product(b)).get_element(0, 0)

                if len(b_v) > 0 and b_v_temp is not None and abs(b_v_new - b_v_temp) < tol:
                    if verbose:
                        print("break at step %d" % i)
                    break

                b_v_temp = b_v_new

            b_v.append(b_v_new)
            b_vec.append(b.transpose().get_list()[0])

            A = A.element_substract(b.mat_product(b.transpose()).scalar_product(b_v_temp))

        self.val, self.vecs = b_v, Matrix(b_vec)

    def fit(self, mat: Matrix):
        self.naive_power_iteration(mat.corr(), verbose=True)

    def transform(self, mat: Matrix):
        assert self.val is not None and self.vecs is not None, "Not fitted"
        return mat.standardize().mat_product(self.vecs.slice(i=(0, self.n_components)).transpose())

    def fit_transform(self, mat: Matrix):
        self.fit(mat)
        return self.transform(mat)


def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])


class KMean:

    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.centroids = None
        self.labels = None

    def init_centroids(self, n: int):
        self.centroids = Matrix([
            [random() for _ in range(n)] for _ in range(self.n_clusters)
        ])

    def get_labels(self, mat: Matrix):
        n = mat.get_shape()[0]

        results = list()
        for i in range(self.n_clusters):
            temp = mat.element_substract(self.centroids.row(i).repeat_vector(n=n))
            temp = temp.element_exponent().sum(axis=0).transpose()
            results.append(temp.get_list()[0])

        return [argmin(r)[0] for r in Matrix(results).transpose().get_list()]

    def new_centroid(self, data, n):
        if len(data) > 0:
            return Matrix(data).mean(axis=1).get_list()[0]
        else:
            return [random() for _ in range(n)]

    def compute_new_centroids(self, n):
        self.centroids = Matrix([
            self.new_centroid([c for c, label in zip(self.centroids, self.labels) if label == i], n=n) for i in range(self.n_clusters)
        ])

    def fit(self, mat: Matrix, n_iter: int=200, tol: float=1e-9, min_iter: int=30, verbose: bool=False):
        s = mat.get_shape()

        self.init_centroids(s[1])

        old_centroids = None
        for i in range(n_iter):

            self.labels = self.get_labels(mat)
            self.compute_new_centroids(s[1])

            if old_centroids is not None and i > min_iter and \
               old_centroids.element_substract(self.centroids).element_exponent().sum() < tol:

                if verbose:
                    print("break at step %d" % i)

                break

            old_centroids = self.centroids

    def transform(self, mat: Matrix):
        return self.get_labels(mat)

    def fit_transform(self, mat: Matrix, verbose: bool=False):
        self.fit(mat, verbose=verbose)
        return self.transform(mat)


if __name__ == "__main__":

    A = Matrix([
        [1, 2, 3, 6],
        [3, 10, 2, 1],
        [4, 9, 1, 1],
        [1, 3, 3, 5],
        [1, 3, 4, 5],
        [0, 3, 3, 5]
    ])

    # PCA

    #pca = PCA()
    #pca.fit(A)
    #print(pca.transform(A).get_list())

    # KMean

    km = KMean(2)

    print(km.fit_transform(A, verbose=True))
