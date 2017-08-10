"""
WELM as given in:
A Robust Indoor Positioning System Based on the
Procrustes Analysis and Weighted Extreme Learning Machine
Paper Authors By: Han Zou, Baoqi Huang, Xiaoxuan Lu, Hao Jiang, Lihua Xie
"""
import numpy as np


class WelmRegressor:
    """
    class for WELM Regressor.
    """

    def __init__(self, train_mat, output_mat, activation_function,
                 num_hidden_neuron, hyper_param_c,weight_mat=None):

        self.train_mat = np.matrix(train_mat)
        self.output_mat = np.matrix(output_mat)

        assert self.train_mat.shape[0] is self.output_mat.shape[0], \
            "input and output should have same shape"

        # number of trainng data is rows of input matrix
        self.num_train_data = self.train_mat.shape[0]

        # number of input neuron is number of columns in input matrix
        self.num_input_neurons = self.train_mat.shape[1]

        # if weight matrix is not given then generate it
        # use contants Weights as diag(1)
        if weight_mat:
            self.weight_mat = np.identity(self.num_input_neurons)
        else:
            self.weight_mat = weight_mat

        # get the number of hidden neurons
        self.num_hidden_neuron = num_hidden_neuron

        # get the activation function
        self.activation_function = activation_function

        # set hyper parameter
        self.hyper_param_c = hyper_param_c

        # build H matrix
        self.h_mat = self.__build_hidden_layer_output_matrix__()

        # build \beta matrix
        self.beta_mat = self.__build_output_weight_matrix__()

    def __build_hidden_layer_output_matrix__(self):
        pass

    def __build_output_weight_matrix__(self):
        pass
