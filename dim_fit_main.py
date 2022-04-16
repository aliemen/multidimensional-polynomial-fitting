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
        return _get_callable(order, params) # _eval_polynomial_order(order, params, *x)

    ZZ = _eval_polynomial_order(order, params, *args)    
    if get_func:
        return ZZ, _get_callable(order, params) # _eval_polynomial_order(order, params, *x)
    else:
        return ZZ



def n_th_derivative(order:list, params:list, index:int, *args, get_func=True, only_func=False, n_th=1):
    '''
    Parameters
    ----------
    order : array_like
        DESCRIPTION.
    params : array_like
        DESCRIPTION.
    index : int
        DESCRIPTION.
    *args : array_like objects (optional)
        Contains the evaluation values for every axis.
    get_func : bool, optional
        Does return a callable function of the n-th derivative if true. The default is True.
    only_func : bool, optional
        Does return only the callable function if true (no otherwise evaluation). The default is False.
    n_th : TYPE, optional
        DESCRIPTION. The default is 1.
    '''
    
    # first, compute the order and params for the derivative
    new_order, new_params = np.array(order), np.array(params)
    for n in range(n_th): # do n_th derivatives
        # get orders and params for derivative, consistency check for index will be done in _first_derivative
        new_order, new_params = _first_derivative(new_order, new_params, index)
    
    # manage returns...
    if only_func:
        return _get_callable(new_order, new_params)     
    
    # evaluate the n_th derivatives at that *args points, consistency checks will be done in _eval_polynomial_order
    ZZ = _eval_polynomial_order(new_order, new_params, *args)
    if get_func:
        return ZZ, _get_callable(new_order, new_params)
    else:
        return ZZ
        
    
    
    


def _get_callable(order:list, params:list):
    # do consistency check
    assert len(order) == len(params), "Number of parameters does not match the summands"
    
    return lambda *x: _eval_polynomial_order(order, params, *x)
        
    
    

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



def _first_derivative(order:list, params:list, index:int):
    '''
    Parameters
    ----------
    order : array_like
        "exponentials_array" that is returned by e.g. "poly_n_fit(...)".
    params : array_like
        "x_params" that is returned by e.g. "poly_n_fit(...)".
    index : int
        Specifies the variable in which the first derivative is to be calculated.
        "index=0" will result in the derivative with respect to the first variable.

    Returns
    -------
    new_order : array_like
        Gives the new "exponentials_array" corresponding to the first derivative.
    new_params : array_like
        Gives the new "x_params" corresponding to the first derivative.
    '''
    
    # consistency check (does this variable exist?)
    assert index < len(order[0]) and 0 <= index, "Index and function dimensions do not match. Please perform derivative with respect to a variable that exists."
    
    # need them as Numpy array (mainly because of the "+=" and scalar-array multiplication)
    order  = np.array(order)
    params = np.array(params)
    
    new_orders = []
    new_params = []
    
    # tmp_param and tmp_order are potentially modified in this loop, keep that in mind
    for tmp_param, tmp_order in zip(params.copy(), order.copy()):
        if tmp_order[index] == 0: # d/dx (constant * x^0) = 0
            new_params.append(0.0)
            new_orders.append(tmp_order)
        else:
            new_params.append(tmp_param*tmp_order[index])
            
            tmp_order[index] = tmp_order[index] - 1
            new_orders.append(tmp_order)
            
    return np.array(new_orders), np.array(new_params)






