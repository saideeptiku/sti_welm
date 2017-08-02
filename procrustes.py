"""
A Robust Indoor Positioning System Based on the
Procrustes Analysis and Weighted Extreme Learning Machine
Paper Authors By: Han Zou, Baoqi Huang, Xiaoxuan Lu, Hao Jiang, Lihua Xie
"""
from scipy.stats import pearsonr
import math
import numpy as np


class STIWELM:
    def __init__(self, train_df, input_labels, output_labels):
        self.train_df = train_df
        self.input_labels = input_labels
        self.output_labels = output_labels

    def get_projected_position(self):
        """
        """
        print(self)


def calculate_sti(vector_a, vector_b):
    """
    calculate the STI value
    STI = [2n(1-p)]^1/2
    p is the pearson correlation
    :return: STI
    """
    assert len(vector_a) == len(vector_b)

    p, _ = pearsonr(vector_a, vector_b)

    n = len(vector_a)

    return math.sqrt(2 * n * (1 - p))


def calculate_sti_each_row(vector_a, vectors):
    """
    calculate the sti for each vector in list
    :param vector_a: vector_a is np 1D array
    :param vectors: 2D array where each row is vector_b
    :return: list of sti values
    """

    for vector_b in vectors:
        # print(vector_b)
        yield calculate_sti(vector_a, vector_b)


# if __name__ == "__main__":
#     s = calculate_sti_each_row(
#         np.array([1, 2, 3]),
#         np.array([[0, 1, 3],
#                   [0, 3, 2]])
#     )
#
#     print(list(s))
