"""
main file for procustes analysis
"""
import procrustes as ps
import pandas as pd
import util_functions as uf
from WELM import WelmRegressor
from WELM import ActFunc
import constants as c
import numpy as np
import matplotlib.pyplot as plt
from CSUDB.data_fetcher import get_map, Place


def get_position(df, index, output_labels):
    return list(np.array(df[output_labels].iloc[index]).flatten())


def sti_welm(train_df, test_df, input_labels, output_labels, at_index=None,
             hyper_param=c.C_HYPER_PARAM, hidden_neurons=c.HIDDEN_LAYER_NEURONS):
    low = 1
    high = test_df.shape[0]

    if at_index:
        low = at_index
        high = low + 1

    position = []
    projected = []

    for i in range(low, high):
        sti = ps.STI(train_df, input_labels, output_labels)

        # print(i, ":", end="")

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

        # print(position[-1], "->", projected[-1], WelmRegressor.aed(position[-1], projected[-1]),flush=True)
        # exit()

    return position, projected


def main():
    """
    main
    """
    # read data training
    train_df = pd.read_csv("CSUDB/CSV/SamsungS6/bc_infill/bc_infill_run1.csv")

    # read validation training
    test_df = pd.read_csv("CSUDB/CSV/oneplus3/bc_infill/bc_infill_run0.csv")

    # make input feature columns list
    input_labels = list(train_df.columns[1:-4])
    # make output feature column list
    output_labels = list(train_df.columns[-3:-1])

    # fill na in columns and keep only required labels
    train_df = uf.fill_na_columns(train_df, input_labels, -100)[input_labels + output_labels]
    test_df = uf.fill_na_columns(test_df, input_labels, -100)[input_labels + output_labels]

    # NA for location
    train_df = train_df.dropna(subset=output_labels)
    test_df = test_df.dropna(subset=output_labels)

    # sensitivity_analysis(train_df, test_df, input_labels, output_labels)
    target, projected = sti_welm(train_df, test_df, input_labels, output_labels)

    aed = WelmRegressor.aed(projected, target, conversion_factor=18)
    print("aed: ", aed)

    eds = WelmRegressor.eds(projected, target, conversion_factor=18)

    print(min(eds), max(eds), np.median(eds))

    plot_on_map(target, projected, get_map(Place.list_all[0]))


def plot_on_map(target, projected, map_path):
    target = np.array(target).reshape(-1, 2)
    projected = np.array(projected).reshape(-1, 2)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    im = plt.imread(map_path)

    implot = plt.imshow(im, origin='upper', extent=[-15, 860, 250, 630])

    real_pos_plt = ax1.scatter(target[:, 0], target[:, 1],
                               c='b', marker='.', alpha=0.6, label="real position")

    # for i, txt in enumerate(range(len(target))):
    #     ax1.annotate(str(txt), (target[i, 0], target[i, 1]), color='blue')

    proj_pos_plt = ax1.scatter(projected[:, 0], projected[:, 1],
                               c='r', marker='.', alpha=0.6, label="projected position")

    # for i, txt in enumerate(range(len(projected))):
    #     ax1.annotate(str(txt), (projected[i, 0], projected[i, 1]), color='red')

    plt.legend(handles=[real_pos_plt, proj_pos_plt])

    # title = TRAIN_DEVICE + " + " + TEST_DEVICE + " + K: " + str(K) + "\n" + "AVG ERROR: " + avg_err

    # plt.title(title)
    plt.show()


def sensitivity_analysis(train_df, test_df, input_labels, output_labels):
    dist = {}

    for x in range(2, 40):
        hp = 1 * (10 ** x)
        # hp = x

        # for hln in range(24, 25) #range(10, 100, 2):
        projected, position = sti_welm(train_df, test_df, input_labels,
                                       output_labels, hyper_param=hp)

        aed = WelmRegressor.aed(projected, position, conversion_factor=18)

        print(hp, ":", aed, flush=True)

        dist[aed] = x

    print("best hyper param: 2 x 10^", dist[min(dist.keys())])


if __name__ == '__main__':
    main()
