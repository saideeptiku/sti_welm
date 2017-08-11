"""
A Robust Indoor Positioning System Based on the
Procrustes Analysis and Weighted Extreme Learning Machine
Paper Authors By: Han Zou, Baoqi Huang, Xiaoxuan Lu, Hao Jiang, Lihua Xie
"""

import math
import constants
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


class STI:
    """
    class for STI
    """

    def __init__(self, train_df, input_labels, output_labels):
        self.train_df = train_df
        self.input_labels = input_labels
        self.output_labels = output_labels

    def get_tds_new(self, test_df, index):
        """
        get the projected position for the data
        given in test data frame at index
        """

        # get test vector
        test_vector = STI.vector_from_df(index, test_df, self.input_labels)

        # get weighted train_df
        train_df = self.filter_by_weight_threshold(test_vector)

        # create new TDS vector
        return self.build_tds_new(train_df, test_vector)

    def build_tds_new(self, train_df, test_vector):
        """
        create the new TDS vector based on equation 16
        """
        sti_values = list(self.sti_each_row_in_df(test_vector, train_df))

        # len of these sti values should be same as number of rows
        assert len(sti_values) == train_df.shape[0]

        # denominator value
        sti_weight_sum = sum([1 / s for s in sti_values])

        # init numerator value
        weight_sub_q_rds = 0
        for sti_sub_q, rds_sub_q in zip(sti_values, train_df[self.input_labels].values.tolist()):
            rds_sub_q = np.array(rds_sub_q)
            weight_sub_q_rds += rds_sub_q * (1 / sti_sub_q)

        return (1 / sti_weight_sum) * (weight_sub_q_rds)

    def filter_by_weight_threshold(self, test_vector):
        """
        create weights column and populate
        remove rows lower than weight threshold
        return df sorted on weights column descending
        """

        # calculate the STI values for test_vector and each data frame
        rp_sti = list(self.sti_each_row_in_df(test_vector, self.train_df))

        # calculate the weights of sti values
        rp_weights = STI.__cal_weights__(rp_sti)

        # attach rp_weights to train_df column and sort by weights
        # first need to check if the number of rows are same as weights
        assert self.train_df.shape[0] == len(rp_weights)

        # keep new dataframe in a temp copy
        train_df = self.train_df.copy()
        train_df[constants.WEIGHT_LABEL] = rp_weights

        # keep all rows where weight is above threshold
        train_df = train_df.loc[train_df[constants.WEIGHT_LABEL]
                                > constants.WEIGHT_THRESHOLD]

        # sort rows in descending order
        train_df = train_df.sort_values(constants.WEIGHT_LABEL,
                                        ascending=False)

        # drop the newly added weight column
        train_df = pd.DataFrame(train_df)
        return train_df.drop(constants.WEIGHT_LABEL, axis=1)

    def sti_each_row_in_df(self, test_vector, data_frame):
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
        sum_sti = sum([1 / s for s in sti_values])

        return [1 / (s * sum_sti) for s in sti_values]

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
        pearson, _ = pearsonr(vector_a, vector_b)

        return math.sqrt(2 * len(vector_b) * (1 - pearson))
