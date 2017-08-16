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

    print(sti.get_tds_new(test_df, 2))


if __name__ == '__main__':
    # main()


    TM = np.matrix([
        [1, 2, 3],
        [6, 5, 8],
        [5, 7, 9]
    ])

    OM = np.matrix([
        [1, 2],
        [3, 4],
        [5, 6]
    ])

    WM = np.matrix([
        [1, 2, 3],
        [6, 5, 8],
        [5, 7, 9]
    ])


    M = WelmRegressor(TM, OM, np.sin, 8, 24, weight_mat=WM)

    PM = np.matrix([
        [1, 2, 1],
        [6, 5, 2],
        [5, 7, 3]
    ])

    print(M.get_projected(PM))
