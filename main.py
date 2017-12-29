"""
main file for procustes analysis
"""
from collections import defaultdict
from WELM import WelmRegressor, ActFunc
from plotter import grouped_places_boxplot_devices
from CSUDB.data_fetcher import Device, Place, get_paths, read_meta, read_csv
import procrustes as ps
import util_functions as uf
import constants as c
import numpy as np
import matplotlib.pyplot as plt

px_to_m = {
    'bc_infill': 20,
    'clark_A': 18,
    'lib': 20,
    'lib_2m': 20,
    'mech_f1': 17
}


def get_position(df, index, output_labels):
    """
    get the values of columns at an index as a list
    use this function to get position
    """
    return list(np.array(df[output_labels].iloc[index]).flatten())


def sti_welm(train_df, test_df, input_labels, output_labels, at_index=None,
             hyper_param=c.C_HYPER_PARAM, hidden_neurons=c.HIDDEN_LAYER_NEURONS):
    """
    base STI-WELM function
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


def sti_wel_csudb(place, train_dev, train_run, test_dev, test_run):
    """
    Wrapper over sti_welm() function to work with CSUDB
    """

    # read data training
    train_paths = get_paths(place, train_dev, train_run, meta=True)
    test_paths = get_paths(place, test_dev, test_run, meta=True)

    # this is because data for one of them is missing
    if not test_paths or not train_paths:
        print("no data for", place, train_dev)
        return [], []

    train_meta_dict = read_meta(train_paths[1])

    training_df = read_csv(train_paths[0], ["WAP*"], ['x', 'y'], replace_na=-100,
                           force_samples_per_rp=5, rename_cols=train_meta_dict)

    # read data testing
    test_paths = get_paths(place, test_dev, test_run, meta=True)
    test_meta_dict = read_meta(test_paths[1])

    validation_df = read_csv(test_paths[0], ["WAP*"], ['x', 'y'], replace_na=-100,
                             force_samples_per_rp=5, rename_cols=test_meta_dict)

    # reverse mac_dict
    mac_dict = dict((v, k) for k, v in train_meta_dict.items())

    # keep common columns
    required_cols = list(set(training_df.columns).intersection(
        set(validation_df.columns)))
    training_df = training_df[required_cols]
    validation_df = validation_df[required_cols]

    # rename columns to WAPS
    training_df = training_df.rename(columns=mac_dict)
    validation_df = validation_df.rename(columns=mac_dict)

    # input/output cols
    input_cols = uf.get_matching_from_list(training_df.columns, "WAP*")
    output_cols = ['x', 'y']

    return sti_welm(training_df, validation_df,
                    input_cols, output_cols)


def do_sti_welm_all(place_list=None, compare_self=True):
    """
    run welm on all devices,
    :returns: dict{place}{dev_train}{dev_test} => list
    """
    errors = defaultdict(lambda: defaultdict(dict))

    if not place_list:
        print("place list not given!")
        place_list = Place.list_all[1:]
    else:
        print(place_list)

    # ignore lg and bc_infill
    for p in place_list:
        for dev_train in Device.list_all[1:]:
            for dev_test in Device.list_all[1:]:

                if dev_train == dev_test and not compare_self:
                    continue

                # print("\n", p, dev_train, dev_test)

                real, guess = sti_wel_csudb(p, dev_train, 0, dev_test, 1)

                error_in_m = [e / px_to_m[p]
                              for e in uf.euclideans(real, guess)]

                print("\n\n", p, dev_train, dev_test,
                      "==>", flush=True, end='')
                try:
                    print(sum(error_in_m) / len(error_in_m), flush=True)
                except:
                    pass

                errors[p][dev_train][dev_test] = error_in_m

    return errors


def write_to_file(filename, ddd_dict):
    """
    write the content of dict to file
    ddd_dict is dict of dict of dict
    last level dict contains list objects
    defaultdict(lambda: defaultdict(dict))
    """
    cols = 'place, device_train, device_test'
    for i in range(100):
        cols += ', err' + str(i)

    f = open(filename, 'w')
    f.write(cols + "\n")
    for p in Place.list_all:
        for d1 in Device.list_all:
            for d2 in Device.list_all:

                # if d1 == d2:
                #     continue

                f.write(p + ", " + d1 + ", " + d2 + ", ")
                try:
                    str_dat = [str(x) for x in ddd_dict[p][d1][d2]]
                except:
                    print(p, d1, d2)
                    str_dat = []
                f.write(", ".join(str_dat) + "\n")
    f.close()


def read_from_file(filename):
    """
    read the content of file to dict
    ddd_dict is dict of dict of dict
    last level dict contains list objects
    defaultdict(lambda: defaultdict(dict))
    """
    errors = defaultdict(lambda: defaultdict(dict))

    lines = []
    with open(filename) as f:
        lines = f.readlines()

    for line in lines[1:]:

        line_list = line.replace(" ", "").strip("\n").split(",")
        p = line_list[0]
        d1 = line_list[1]
        d2 = line_list[2]

        data_list = list(line_list[3:])

        if data_list:
            data = [float(x) for x in data_list[3:]]
        else:
            data = []
            print()

        errors[p][d1][d2] = data

    return errors


def plot_on_map(target, projected, map_path, extent=None):
    """
    plot target and projected data on a map at given path
    extent should be a list as in imshow()
    """
    target = np.array(target).reshape(-1, 2)
    projected = np.array(projected).reshape(-1, 2)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    im = plt.imread(map_path)

    if extent:
        plt.imshow(im, origin='upper', extent=extent)
    else:
        plt.imshow(im, origin='upper', extent=[-15, 860, 250, 630])

    real_pos_plt = ax1.scatter(target[:, 0], target[:, 1],
                               c='b', marker='.', alpha=0.6,
                               label="real position")

    # for i, txt in enumerate(range(len(target))):
    #     ax1.annotate(str(txt), (target[i, 0], target[i, 1]), color='blue')

    proj_pos_plt = ax1.scatter(projected[:, 0], projected[:, 1],
                               c='r', marker='.', alpha=0.6,
                               label="projected position")

    # for i, txt in enumerate(range(len(projected))):
    #     ax1.annotate(str(txt), (projected[i, 0], projected[i, 1]), color='red')

    plt.legend(handles=[real_pos_plt, proj_pos_plt])

    # title = TRAIN_DEVICE + " + " + TEST_DEVICE + " + K: " + str(K) + "\n" + "AVG ERROR: " + avg_err

    # plt.title(title)
    plt.show()


def sensitivity_analysis(train_df, test_df, input_labels, output_labels):
    """
    sensitivuty analysis on hyper-parameter
    """
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


def main():
    """
    main
    """
    # write_to_file('results.csv', do_sti_welm_all(place_list=[Place.lib_2m]))
    # do_sti_welm_all(compare_self=True)
    # for dev in Device.list_all[1:]:
    #     grouped_places_boxplot_devices(read_from_file('results/results.csv'), train_device=Device.oneplus2)
    grouped_places_boxplot_devices(read_from_file('results.csv'),
                                   train_device=Device.samsung_s6,
                                   test_devices=[
                                       Device.oneplus2, Device.samsung_s6, Device.oneplus3],
                                   places=[Place.lib_2m, Place.clark_a, Place.mech_f1])


if __name__ == '__main__':
    main()
