"""The combination of STI (Procrustes Analysis) and WELM Regression"""
from WELM import ActFunc
import constants as c
from WELM import WelmRegressor
import numpy as np
import procrustes as ps
from WELM import WelmRegressor

euclidean_distances = WelmRegressor.eds
mean_euclidean_distance = WelmRegressor.aed


def get_position(df, index, output_labels):
    return list(np.array(df[output_labels].iloc[index]).flatten())


def sti_welm(train_df, test_df, input_labels, output_labels, at_index=None,
             hyper_param=c.C_HYPER_PARAM, hidden_neurons=c.HIDDEN_LAYER_NEURONS):
    """
    Project positions fot test_df based on the STI_WELM technique using train_df.
    :param train_df: no NANs should be present
    :param test_df: no NANs should be present
    :param input_labels: list of input labels present in test and train
    :param output_labels: list of input labels present in test and train
    :param at_index: get projected output for input at specific index, start at 1
    :param hyper_param: value of constant hyper-parameter, found empirically
    :param hidden_neurons: number of neurons per hidden layer
    :return: list of actual position and projected position
    """

    low = 1
    high = test_df.shape[0]

    if at_index:
        low = at_index
        high = low + 1

    position = []
    projected = []

    for i in range(low, high):
        sti = ps.STI(train_df, input_labels, output_labels)

        tds = sti.get_tds_new(i, test_df=test_df)

        if tds is None:
            continue

        # should be M x M
        W = sti.get_weight_matrix_for_welm(sample_per_ref_point=1)

        in_mat, out_mat = sti.get_rds_new()

        M = WelmRegressor(in_mat, out_mat, ActFunc.hardlim,
                          hidden_neurons, hyper_param, weight_mat=W)

        position.append(get_position(test_df, i, output_labels))
        projected.append(list(np.array(M.get_projected(tds)).flatten()))

    return position, projected
