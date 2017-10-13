"""
Procrustes analysis as given in:
A Robust Indoor Positioning System Based on the
Procrustes Analysis and Weighted Extreme Learning Machine
Paper Authors By: Han Zou, Baoqi Huang, Xiaoxuan Lu, Hao Jiang, Lihua Xie
"""

import math
import constants
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings

class STI:
    """
    class for STI
    """

    def __init__(self, train_df, input_labels, output_labels):
        self.train_df = train_df
        self.input_labels = input_labels
        self.output_labels = output_labels

        self.sti_weight_list_q = []
        self.train_df_thrsh = None

    def get_tds_new(self, index_or_vector, test_df=None):
        """
        provide test vector itself
        test_df required when index provided
        index begins with 1
        """

        # get test vector
        if isinstance(index_or_vector, list):
            test_vector = index_or_vector
        else:
            test_vector = STI.vector_from_df(index_or_vector, test_df, self.input_labels)

        # get weighted train_df
        # RDSq
        self.train_df_thrsh = self.__filter_by_weight_threshold__(test_vector)

        if self.train_df_thrsh is None:
            print("No rows above threshold for", index_or_vector)
            print("tds_new could not be formed")
            return None
        else:
            pass
            # print("Q (train rows above(", str(constants.WEIGHT_THRESHOLD), "):", self.train_df_thrsh.shape[0])

        # create new TDS vector
        return self.__build_tds_new__(self.train_df_thrsh, test_vector)

    def get_rds_new(self):
        """
        get all values that are greater than threshold values
        returns two matrices of input and target with same number of rows
        """

        if self.train_df_thrsh is None:
            exit("run get_tds_new first!")

        input_mat = np.matrix(self.train_df_thrsh.drop(self.output_labels, axis=1).as_matrix())

        output_mat = np.matrix(self.train_df_thrsh[self.output_labels])

        assert input_mat.shape[0] == output_mat.shape[0], "shape mismatch!"

        return input_mat, output_mat

    def get_weight_matrix_for_welm(self, sample_per_ref_point):
        """
        get the weight matrix (W) as described in the paper.
        """

        if not self.sti_weight_list_q:
            exit("run get_tds_new first!")

        diag_mat = np.diag(self.sti_weight_list_q)

        i_f = np.identity(sample_per_ref_point)

        return (1 / sum(self.sti_weight_list_q)) * np.kron(diag_mat, i_f)

    def __build_tds_new__(self, train_df, test_vector):
        """
        create the new TDS vector based on equation 16
        """
        sti_values = list(self.__sti_each_row_in_df__(test_vector, train_df))

        # len of these sti values should be same as number of rows
        assert len(sti_values) == train_df.shape[0]

        # denominator value
        sti_weight_sum = sum([1 / s if s > 0 else constants.INF for s in sti_values])

        # init numerator value
        weight_sub_q_rds = 0
        for sti_sub_q, rds_sub_q in zip(sti_values, train_df[self.input_labels].values.tolist()):
            rds_sub_q = np.array(rds_sub_q)

            try:
                weight_sub_q_rds += rds_sub_q * (1 / sti_sub_q)
            except ZeroDivisionError:
                weight_sub_q_rds += rds_sub_q * constants.INF

            try:
                self.sti_weight_list_q.append((1 / sti_sub_q))
            except ZeroDivisionError:
                self.sti_weight_list_q.append(constants.INF)

        tds_new = weight_sub_q_rds / sti_weight_sum

        return np.array(tds_new).reshape(1, -1)

    def __filter_by_weight_threshold__(self, test_vector):
        """
        create weights column and populate
        remove rows lower than weight threshold
        return df sorted on weights column descending
        :rtype: pandas dataframe, None
        """

        # calculate the STI values for test_vector and each data frame
        rp_sti = list(self.__sti_each_row_in_df__(test_vector, self.train_df))

        # calculate the weights of sti values
        rp_weights = STI.__cal_weights__(rp_sti)

        # attach rp_weights to train_df column and sort by weights
        # first need to check if the number of rows are same as weights
        assert self.train_df.shape[0] == len(rp_weights)

        # keep new dataframe in a temp copy
        train_df = self.train_df.copy()
        train_df[constants.WEIGHT_LABEL] = rp_weights

        pre_rows = train_df.shape[0]

        # keep all rows where weight is above threshold
        train_df = train_df.loc[train_df[constants.WEIGHT_LABEL]
                                > constants.WEIGHT_THRESHOLD]

        if train_df.empty and pre_rows > 0:
            print("No rows above given threshold.")
            return None

        # sort rows in descending order
        train_df = train_df.sort_values(constants.WEIGHT_LABEL,
                                        ascending=False)
        # drop the newly added weight column
        train_df = pd.DataFrame(train_df)
        return train_df.drop(constants.WEIGHT_LABEL, axis=1)

    def __sti_each_row_in_df__(self, test_vector, data_frame):
        """
        calculate the sti for each vector in dataframe
        :param vector_a: vector_a is np 1D array
        :param vectors: 2D array where each row is vector_b
        :return: list of sti values
        """
        # calculate the STI values for test_vector and each data frame
        for _, row in data_frame[self.input_labels].iterrows():
            yield STI.calculate_sti(list(row), test_vector)

    @staticmethod
    def __cal_weights__(sti_values):
        sum_sti = sum([(1 / s) if s > 0 else constants.INF for s in sti_values])

        return [1 / (s * sum_sti) if s > 0 else constants.INF for s in sti_values]

    @staticmethod
    def vector_from_df(index, dataframe, input_labels):
        """
        get RSS values as vector at index with position labels in list
        :param df: data frame pandas
        :param index: starts from 0
        :param input_labels: list of labels
        :return:
        """

        return list(dataframe[input_labels].iloc[index])

    @staticmethod
    def calculate_sti(vector_a, vector_b):
        """
        calculate the STI value
        STI = [2n(1-p)]^1/2
        p is the pearson correlation
        :return: STI
        """
        assert len(vector_a) == len(vector_b)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            pearson, _ = pearsonr(vector_a, vector_b)

        if isNan(pearson):
            pearson = 0

        sti = math.sqrt(2 * len(vector_b) * (1 - pearson))

        return sti

def isNan(num):
    return num != num
