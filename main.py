"""
main file for procustes analysis
"""
# import matplotlib.pyplot as plt
# import csv
import procrustes as ps
import pandas as pd
import util_functions as uf
# from label_x_y_locations import label_similar_locations
import numpy as np
from WELM import WelmRegressor
import constants as c


def main():
    """
    main
    """
    # read data training
    train_df = pd.read_csv("data/bcinfill-s6-run1.csv")

    # read validation training
    test_df = pd.read_csv("data/bcinfill-s6-run2.csv")

    # make input feature columns list
    input_labels = list(train_df.columns[1:-4])
    # make output feature column list
    output_labels = list(train_df.columns[-3:-1])

    # fill na in columns and keep only required labels
    train_df = uf.fillna_in_columns(
        train_df, input_labels, -100)[input_labels + output_labels]
    test_df = uf.fillna_in_columns(
        test_df, input_labels, -100)[input_labels + output_labels]

    train_df = train_df.dropna(subset=output_labels)
    test_df = test_df.dropna(subset=output_labels)

    sti = ps.STI(train_df, input_labels, output_labels)

    tds = sti.get_tds_new(2, test_df=test_df)

    # should be M x M
    W = sti.get_weight_matrix_for_welm(1)

    print(W.shape)

    in_mat, out_mat = sti.get_rds_new()

    print(in_mat.shape, out_mat.shape)

    M = WelmRegressor(in_mat, out_mat, np.sin, c.HIDDEN_LAYER_NEURONS, c.C_HYPER_PARAM)

    print(tds.shape, in_mat.shape, out_mat.shape)

    M.get_projected(tds)


if __name__ == '__main__':
    main()
