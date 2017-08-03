# import modules
import pandas as pd
from label_x_y_locations import label_similar_locations
from util_functions import print_df, fillna_in_columns
import matplotlib.pyplot as plt
import csv
import procrustes as ps


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
    train_df = fillna_in_columns(train_df, input_labels, -100)[input_labels + output_labels]
    test_df = fillna_in_columns(test_df, input_labels, -100)[input_labels + output_labels]

    sti = ps.STIWELM(train_df, input_labels, output_labels)

    sti.get_projected_position(test_df, 2)


if __name__ == '__main__':
    main()
