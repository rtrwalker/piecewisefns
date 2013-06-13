# piecewisefns - classes and functions to manipulate piecewise functions
# Copyright (C) 2013 Rohan T. Walker (rtrwalker@gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/gpl.html.

"""
module for piecewise 1d linear relationships

"""
from __future__ import print_function, division

import numpy as np


def has_steps(x):
    """check if data points have any step changes
    
    True if any two consecutive x values are equal
    
    Parameters
    ----------
    x : array_like
        x-coordinates
    y : array_like
        y-coordinates 
        
    Returns
    -------
    out : boolean
        returns true if any two consecutive x values are equal
        
    """
    x = np.asarray(x)
    #y = np.asarray(y)
    return np.any(np.diff(x)==0)
    

def is_initially_increasing(x):
    """Are first two values increasing?
    
    finds 1st instance where x[i+1] != x[i] and checks if x[i+1] > x[i]
    
    Parameters
    ----------
    x : array_like
        1 dimensional data to check
        
    Returns
    -------
    out : ``int``
        returns True is if 2nd value is greater than the 1st value
        returns False if 2nd value is less than the 1st value
        
    """
    
    #this might be slow for long lists, perhaps just loop through until x[i+1]!=x[i]
    if x[1]!=x[0]:
        return x[1]>x[0]
    x = np.asarray(x)
    return np.where(np.diff(x)!=0)[0][0]>0
    
        


#used info from http://stackoverflow.com/questions/4983258/python-how-to-check-list-monotonicity
def strictly_increasing(x):
    """Checks all x[i+1] > x[i]"""
    x = np.asarray(x)
    return np.all(np.diff(x)>0)

def strictly_decreasing(x):
    """Checks all x[i+1] < x[i]"""
    x = np.asarray(x)
    return np.all(np.diff(x)<0)

def non_increasing(x):
    """Checks all x[i+1] <= x[i]"""
    x = np.asarray(x)
    return np.all(np.diff(x)<=0)

def non_decreasing(x):
    """Checks all x[i+1] >= x[i]"""
    x = np.asarray(x)
    return np.all(np.diff(x)>=0)



def non_increasing_and_non_decreasing_parts(x, include_end_point = False):
    """split up a list into sections that are non-increasing and non-decreasing
    
    Returns
    -------
    
    A : 2d array
        each element of A is a list of the start indices of each line segment that is part of a particular 
        non-increasing or non-decreasing run
    
    Notes
    -----
    This funciton only returns start indecies for each line segment. 
    Lets say x is [0,4 , 5.5] then A will be [[0,1]].  If you do x[A[0]] then 
    you will get [0, 4] i.e. no end point. To get the whole increasing or 
    decreasing portion including the end point you need to do something like 
    x[A[0].append[A[0][-1]+1]].
    """
    
    x = np.asarray(x)
    sign_changes = np.sign(np.diff(x))
    A = [[0]]
    
    current_sign = sign_changes[np.where(sign_changes!=0)[0][0]]
    
    for i, sgn in enumerate(sign_changes.tolist()):
        if i == 0: 
            continue
        if sgn != 0 and sgn != current_sign:
            if include_end_point:
                A[-1].append(i)
            A.append([])
            current_sign = sgn
        A[-1].append(i)
    if include_end_point: #append final end point
        A[-1].append(A[-1][-1] + 1)    
    return A
    
def force_strictly_increasing(x, y = None, keep_end_points = True, eps = 1e-15):
    """force a non-decreasing or non-increasing list into a strictly increasing
    
    Adds or subtracts tiny amounts (multiples of `eps`) from the x values in 
    step changes to ensure no two consecutive x values are equal (i.e. make x 
    strictly increasing).  The adjustments are small enough that for all 
    intents and purposes the data behaves as before; it can now however be 
    easily used in straightforward interpolation functions that require 
    strictly increasing x data.
    
    Any data where x is non-increasing will be reversed.
    
    Parameters
    ----------
    x : array_like
        list of x coordinates
    y : array_like, optional
        list of y coordinates, (default = None, if known that x is 
        non-decreasing then y will not be affected)
    keep_end_points : ``boolean``, optional
        determines which x value of the step change is adjusted.  
        Consider x=[1,1] and y=[20,40]. If keep_end_points==True then new 
        data will be x=[0.9999,1], y=[20,40].  If keep_end_points==False then 
        data will be x=[1, 0.9999], y=[20,40]
    eps : float, optional
        amount to add/subtract from x (default is 1e-15).  To ensure 
        consecutive step changes are handled correctly multipes of `eps` 
        will be added and subtracted. e.g. if there are a total of five 
        steps in the data then the first step would get 5*`eps` adjustment, 
        the second step 4*`eps` adjustment and so on.
        
    """
    
    x = np.asarray(x)
    y = np.asarray(y)    
    
    if strictly_increasing(x):
        return x, y
        
    if strictly_decreasing(x):
        return x[::-1], y[::-1]
    
    if non_increasing(x):
        x = x[::-1]
        y = y[::-1]
        
    if not non_decreasing(x):
        raise ValueError, "x data is neither non-increasing, nor non-decreasing, therefore cannot force to strictly increasing"
        
            
    steps = np.where(np.diff(x) == 0)[0]    
    
    if keep_end_points:
        f = -1 * eps
        d = 0
        dx = np.arange(len(steps),0, -1) * f
    else:
        f = 1 * eps
        d = 1
        dx = np.arange(1,len(steps)+1) * f
            
    x[steps + d] = x[steps + d] + dx
    return x, y

def force_non_decreasing(x, y=None):
    """force non-increasing x, y data to non_decreasing by reversing the data
    
    Leaves already non-decreasing data alone.
    
    Parameters
    ----------
    x, y: array_like
        x and y coordinates
        
    Returns
    -------
    x,y : ndarray, ndarray
        non-decreasing version of x, y
        
    """
    x = np.asarray(x)
    y = np.asarray(y)    
    
    if non_decreasing(x):
        return x, y
    
    if not non_increasing(x):
        raise ValueError, "x data is neither non-increasing, nor non-decreasing, therefore cannot force to non-decreasing"        
                    
    return x[::-1], y[::-1]    
    
        
    
    
    
    
    
def ramps_constants_steps(x, y):
    """find the ramp segments, constant segments and step segments in x, y data
    
    returns a tuple of lists indicating the start indecies for all ramp, 
    constant and step segments in `x`, `y` data.  Assumes data is 
    non_decreasing.
    
    Parameters
    ----------
    x, y : array_like
        x and y coords (must be non-decreasing)
        
    Returns
    -------
    ramps : array_like
        start indecies of all ramps. e.g. of a ramp is 
        x=[0,2], y=[10,15]
    constants : array_like
        start indecies of all constant sections. e.g. of a constant section is 
        x=[0,2], y=[15,15]
    steps : array_like
        start indecies of all step. e.g. of a step is 
        x=[1,1], y=[5,15]
        
    """
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    #find start index of instantaneous loads
    steps = np.where(np.diff(x)==0)[0]    
    
    #find start index of constant loads
    constants = np.where(np.diff(y)==0)[0]
    
    #find start index of ramp loads    
    ramps = np.delete(np.arange(len(x)-1),np.concatenate((steps, constants)))                

    return ramps, constants, steps

def start_index_of_ramps(x, y):
    """find the start indecies of the ramp segments in x, y data.
    
    An example of a 'ramp' x=[0,2], y=[10,15]. i.e. not a vertical line and 
    not a horizontal line.
    
    Assumes data is non_decreasing.
    
    Parameters
    ----------
    x, y : array_like
        x and y coords (must be non-decreasing)
        
    Returns
    -------
    out : array_like
        start indecies of all ramps. 
        
    """
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    
    #return np.where(np.diff(x)!=0 & np.diff(y)!=0)[0]    
    return  np.where((np.diff(x)!=0) & (np.diff(y)!=0))[0]
    
def start_index_of_constants(x, y):
    """find the start indecies of the constant segments in x, y data.
    
    An example of a 'ramp' x=[0,2], y=[15,15]. i.e. a horizontal line
    
    Assumes data is non_decreasing.
    
    Segments such as x=[1,1], y=[2,2] are ignored.
    
    Parameters
    ----------
    x, y : array_like
        x and y coords (must be non-decreasing)
        
    Returns
    -------
    out : array_like
        start indecies of all constant segments. 
        
    """
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    
    return np.delete(np.where(np.diff(y)==0)[0], np.where((np.diff(x)==0) & (np.diff(y)==0))[0])   

def start_index_of_steps(x, y):
    """find the start indecies of the step segments in x, y data.
    
    An example of a 'step' x=[0,0], y=[10,15]. i.e. a vertical line
    
    Assumes data is non_decreasing.
    
    Segments such as x=[1,1], y=[2,2] are ignored.
    
    Parameters
    ----------
    x, y : array_like
        x and y coords (must be non-decreasing)
        
    Returns
    -------
    out : array_like
        start indecies of all step segments. 
        
    """
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    
    return np.delete(np.where(np.diff(x)==0)[0], np.where((np.diff(x)==0) & (np.diff(y)==0))[0])

def ramps_constants_steps_after(x,y,xi):
    """find the ramp segments, constant segments and step segments in x, y data that start after certina x values
    
    returns a tuple of lists indicating the start indecies for all ramp, 
    constant and step segments in `x`, `y` data.  Assumes data is 
    non_decreasing.
    
    Parameters
    ----------
    x, y : array_like
        x and y coords (must be non-decreasing)
        
    Returns
    -------
    ramps : array_like
        start indecies of all ramps. e.g. of a ramp is 
        x=[0,2], y=[10,15]
    constants : array_like
        start indecies of all constant sections. e.g. of a constant section is 
        x=[0,2], y=[15,15]
    steps : array_like
        start indecies of all step. e.g. of a step is 
        x=[1,1], y=[5,15]
        
    """

    
if __name__ == '__main__':
    #print(strictly_increasing([0,  0.5,  1,  1.5,  2]))
    #print(non_increasing_and_non_decreasing_parts([0,  0.5,  1,  1.5,  2]))
    #print (force_strictly_increasing([0, 0.5, 1, 0.75, 1.5, 2], keep_end_points=True))
   # print(force_strictly_increasing([0,  0.4,   0.4,  1,  2.5,  3,  3], keep_end_points = True, eps=0.01) )
    #print(force_strictly_increasing([0,  0.4,   0.4,  1,  2.5,  3,  3], keep_end_points = False, eps=0.01) )
    #print (force_strictly_increasing([0,  0.4,   0.4,  1,  2.5,  3,  3]))
    a = {'x': [0,  0,  1,  1,  2],                          'y': [0, 10, 10, 30, 30]}
    a = {'x': [0,  0.4,  1,  1.5,  2],                          'y': [0, 10, 10, 30, 30]}
#    print(ramps_constants_steps(**a))
    print (start_index_of_ramps(**a))   
    
#import sys
#sys.float_info.epsilon


#def passes_vertical_line_test(x, y):
