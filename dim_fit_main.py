# -*- coding: utf-8 -*-
'''
Created on Fri Apr 15 08:49:14 2022
Copyright (C) 2022  https://github.com/aliemen/

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''



import numpy as np
from scipy.optimize import leastsq

import itertools


def poly_n_val(order, params, *args, get_func=True, only_func=False):    
    if only_func:
        return lambda *x: _eval_polynomial_order(order, params, *x)

    ZZ = _eval_polynomial_order(order, params, *args)    
    if get_func:
        return ZZ, lambda *x: _eval_polynomial_order(order, params, *x)
    else:
        return ZZ
    
    

def _eval_polynomial_order(order:list, params:list, *args):
    '''
    Parameters
    ----------
    order : array_like
        Obtained array from _get_exponential_indizes containing all exponent combinations
    *args : array_like objects or floats
        Contains the evaluation values for every axis

    Returns
    -------
    A matrix with len(args) axis containing all evaluated elements.
    '''
    assert len(order) == len(params), "Number of parameters does not match the summands"
    assert len(order[0]) == len(args), "Number of arguments does not match expected number of axes."
    
    single_point_eval = isinstance(args[0], float) or isinstance(args[0], int)
    for tmp_axis in args:
        assert (isinstance(tmp_axis, float) or isinstance(tmp_axis, int)) == single_point_eval, "Evaluation need only arrays or only floats!"
        
    # in this case, just return that one float multiplied with all indizes
    if single_point_eval:
        ret_value = 0
        for tmp_order, tmp_param in zip(order, params):
            tmp_mult = 1
            for i, x_i in zip(tmp_order, args):
                tmp_mult *= x_i**i
            ret_value += tmp_param*tmp_mult
        
        return ret_value
     
    
    ### ----- else move on... ----- ###
    #n_axis = len(args) # <-- not used for now
    
    # build the shape for the return matrix
    return_shape = (len(args[0]),)
    for tmp_arg in args[1:]:
        return_shape += (len(tmp_arg),)
    
    ret_matrix = np.zeros(return_shape)
    
    variable_meshgrids = np.meshgrid(*args, indexing='ij')
    
    for tmp_param, tmp_ind_tupel in zip(params, order):
        tmp_result = np.ones_like(ret_matrix)
        
        # multiply polynomial in every dimension together
        for tmp_index, tmp_grid in zip(tmp_ind_tupel, variable_meshgrids):
            tmp_result *= tmp_grid**tmp_index
        
        # add polynomials to the final one
        ret_matrix += tmp_param*tmp_result
    
    return ret_matrix
    

def poly_n_fit(func_values, max_orders:list, *args, max_total_order=None):
    '''
    Parameters
    ----------
    func_values : array_like
        Function vaues to which the fit is to be performed. Attention: There are two different indexing methods in np.meshgrid(). This 
        function uses indexing="ij", which means that func_values[i,j] corresponds two the first axis i and the second j. If we have
        a function evaluated over a normal "np.meshgrid(x, y)", the two indizes are changed. That means, the order of *args must be
        x1, x2, ..., which is possibly different to just x, y, ...
        Therefore, please adapt the order of *args to the axis of func_values!
    max_orders : array_like
        Max polynomial order for every axis.
    max_total_order : int
        Max order for the whole polynomial
    *args : n_axis array_like objects
        x1, x2, x3, ... axist data.

    Returns
    -------
    Returns the fitted parameters together with the exponent-"combinations", which allows to reconstruct the polynomial (for example for "poly_n_val")
    '''
    
    #ensures only a numpy array is used
    func_values = np.array(func_values)
    
    # gives the "n" (number of dimensions in which to fit)
    input_data_shape = func_values.shape
    n_axis           = len(input_data_shape)
    #print(input_data_shape)
    # consistency check of the input data #
    assert len(max_orders) == len(args), "The array containing the orders for every axis is not well defined!"
    for i in range(n_axis):
        #args[i] = np.array(args[i]) # we want it as a numpy array!
        
        # should only contain one axis!
        assert input_data_shape[i] == args[i].shape[0], "Function values do not match given axis values! Please use \"indexing='ij'\" in Meshgrid!"
        
        
    # stores the number of indizes of the polynomial for every axis
    exponentials_array = _get_exponential_indizes(max_orders, max_total_order)
    
    # give some start parameters
    x0_params = np.ones(len(exponentials_array))
    
    # gives a "difference function" that returns an n-dimensional "matrix"
    def F(params:list):
        return _eval_polynomial_order(exponentials_array, params, *args) - func_values
    
    
    x_params, cov_x, infodict, mesg, ier = leastsq(lambda p: F(p).flatten(), x0_params, full_output=True)
    print("Residual Norm =", np.linalg.norm(F(x_params)))
    #print(infodict, ier)
    return x_params, exponentials_array
    
    
        
    
def _get_exponential_indizes(max_orders, max_total_order):
    # get all possible exponents combinations
    max_index              = max(max_orders)
    #print(max_index)
    n_axis                 = len(max_orders)
    #all_index_combinations = np.array([seq for seq in itertools.permutations(range(max_index+1), n_axis)])
    all_index_combinations = np.array([seq for seq in itertools.product(range(max_index+1), repeat=n_axis)])
    
    # lastly, we check which exponents are suitable and use the order for every evaluation later
    used_order = []
    if max_total_order is None:
        used_order = all_index_combinations
    else:
        for index_tupel in all_index_combinations:
            # if the total index is not to big, we add the tupel
            if sum(index_tupel) <= max_total_order:
                used_order.append(index_tupel)
        
    return np.array(used_order)



