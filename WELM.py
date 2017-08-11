"""
WELM as given in:
A Robust Indoor Positioning System Based on the
Procrustes Analysis and Weighted Extreme Learning Machine
Paper Authors By: Han Zou, Baoqi Huang, Xiaoxuan Lu, Hao Jiang, Lihua Xie
"""
import numpy as np
import math


class WelmRegressor:
    """
    class for WELM Regressor.
    """

    def __init__(self, train_mat, output_mat, activation_function,
                 num_hidden_neuron, hyper_param_c, weight_mat=None):

        self.train_mat = np.matrix(train_mat)
        self.t_mat = np.matrix(output_mat)

        assert self.train_mat.shape[0] is output_mat.shape[0], \
            "input and output should have same shape"

        # number of trainng data is rows of input matrix
        self.num_train_data = self.train_mat.shape[0]

        # number of input neuron is number of columns in input matrix
        self.num_input_neurons = self.train_mat.shape[1]

        # if weight matrix is not given then generate it
        # use contants Weights as diag(1)
        if weight_mat is None:
            self.w_mat = np.identity(self.num_input_neurons)
        else:
            self.w_mat = weight_mat

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

        # trained output
        # this matrix should be cloose to training data
        self.trained_output_mat = self.h_mat * self.beta_mat

        # get the output accuracy
        print("Trained with output accuracy: ", self.get_trained_accuracy())
        # get the output accuracy
        print("Trained with AED: ", self.get_trained_average_distance())

    def get_trained_accuracy(self):
        """
        get the RMSE value of the trained model
        """
        return WelmRegressor.rmse(self.trained_output_mat, self.t_mat)

    def get_trained_average_distance(self):
        """
        get the average euclidean distance between model created and training data
        """
        return WelmRegressor.aed(self.trained_output_mat, self.t_mat)


    @staticmethod
    def aed(predictions, targets):
        """
        get the average eucledian distance between prediction and target
        """
        return np.sum(np.square(predictions - targets), axis=1).mean()

    @staticmethod
    def rmse(predictions, targets):
        """
        get the RMSE between prediction and target
        based on sklearn.metrics.mean_squared_error
        """
        out_errors = np.average(np.square((predictions - targets)), axis=0)
        return np.average(out_errors) 

    def __build_hidden_layer_output_matrix__(self):
        # between -1 to 1
        input_weight = np.random.rand(
            self.num_hidden_neuron, self.num_input_neurons) * 2 - 1

        bias_of_hidden_neurons = np.random.rand(self.num_hidden_neuron, 1)

        temp_h = input_weight * self.train_mat

        # shape of H matrix and bias matrix should match
        # create two more coloumns fo ones
        ones = np.ones(
            [
                bias_of_hidden_neurons.shape[0],
                self.num_train_data - bias_of_hidden_neurons.shape[1]
            ], )
        bias_matrix = np.append(bias_of_hidden_neurons, ones, axis=1)

        temp_h = temp_h + bias_matrix

        # the matrix in paper is actually
        # the transpose of the matrix created
        temp_h = np.transpose(temp_h)

        return self.activation_function(temp_h)

    def __build_output_weight_matrix__(self):
        if self.num_train_data < self.num_hidden_neuron:
            return self.__beta_l_is_greater_than_m__()
        else:
            return self.__beta_m_is_greater_than_l__()

    def __beta_m_is_greater_than_l__(self):
        inner_mat = self.h_mat.transpose() * self.w_mat * self.h_mat

        i_by_c = np.identity(inner_mat.shape[0]) / self.hyper_param_c

        sum_inner_c_mat = inner_mat + i_by_c

        outter_mat = self.h_mat.transpose() * self.w_mat * self.t_mat

        return np.linalg.lstsq(sum_inner_c_mat, outter_mat)[0]

    def __beta_l_is_greater_than_m__(self):

        inner_mat = self.w_mat * self.h_mat * self.h_mat.transpose()

        i_by_c = np.identity(self.t_mat.shape[0]) / self.hyper_param_c

        sum_inner_c_mat = inner_mat + i_by_c

        outter_mat = self.w_mat * self.t_mat

        inv_mul_mat = np.linalg.lstsq(sum_inner_c_mat, outter_mat)[0]

        return self.h_mat.transpose() * inv_mul_mat


# if __name__ == "__main__":

#     TM = np.matrix([
#         [1, 2, 3],
#         [6, 5, 8],
#         [5, 7, 9]
#     ])

#     OM = np.matrix([
#         [1, 2],
#         [3, 4],
#         [5, 6]
#     ])

#     WelmRegressor(TM, OM, np.sin, 2, 24)
