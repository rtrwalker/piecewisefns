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
from __future__ import division, print_function

from nose import with_setup
from nose.tools.trivial import assert_almost_equal
from nose.tools.trivial import assert_raises
from nose.tools.trivial import ok_
from nose.tools.trivial import assert_false
from nose.tools.trivial import assert_equal

from math import pi
import numpy as np

from piecewisefns.piecewise_linear_1d import strictly_decreasing
from piecewisefns.piecewise_linear_1d import strictly_increasing
from piecewisefns.piecewise_linear_1d import non_decreasing
from piecewisefns.piecewise_linear_1d import non_increasing
from piecewisefns.piecewise_linear_1d import is_initially_increasing
from piecewisefns.piecewise_linear_1d import has_steps
from piecewisefns.piecewise_linear_1d import non_increasing_and_non_decreasing_parts
from piecewisefns.piecewise_linear_1d import force_strictly_increasing
from piecewisefns.piecewise_linear_1d import force_non_decreasing
from piecewisefns.piecewise_linear_1d import start_index_of_steps
from piecewisefns.piecewise_linear_1d import start_index_of_ramps
from piecewisefns.piecewise_linear_1d import start_index_of_constants
from piecewisefns.piecewise_linear_1d import ramps_constants_steps

class test_linear_piecewise(object):
    """Some piecewise distributions for testing"""
        
    def __init__(self):
        self.two_steps = {'x': [0,  0,  1,  1,  2],
                          'y': [0, 10, 10, 30, 30]}
        self.two_steps_reverse = {'x': [0,  0,  -1,  -1,  -2],
                                  'y': [0, 10,  10,  30,  30]}
        self.two_ramps = {'x': [0,  0.5,  1,  1.5,  2],
                          'y': [0, 10, 10, 30, 30]}
        self.two_ramps_reverse = {'x': [0,  -0.5,  -1,  -1.5,  -2],
                                  'y': [0,  10.0,  10,  30.0,  30]}
        self.two_ramps_two_steps = {'x': [0,  0.4,   0.4,  1,  2.5,  3,  3],
                                    'y': [0,  10.0, 20.0, 20, 30.0, 30, 40]}
        self.two_ramps_two_steps_reverse = {'x': [0,  -0.4,   -0.4,  -1,  -2.5,  -3,  -3],
                                            'y': [0,  10.0, 20.0, 20, 30.0, 30, 40]}
        self.switch_back = {'x': [0, 0.5, 1, 0.75, 1.5, 2],
                            'y': [0, 1.2, 2, 2.25, 3.5, 3]} 
        self.switch_back_steps = {'x': [0, 0, 1, 0.75, 0.75, 2],
                            'y': [0, 1.2, 2, 2.25, 3.5, 3]}                                             
                                    
    def test_has_steps(self):
        """test some has_steps examples"""
        
        ok_(has_steps(self.two_steps['x']))
        ok_(has_steps(self.two_steps_reverse['x']))
        assert_false(has_steps(self.two_ramps['x']))
        assert_false(has_steps(self.two_ramps_reverse['x']))        
        ok_(has_steps(self.two_ramps_two_steps['x']))
        ok_(has_steps(self.two_ramps_two_steps_reverse['x']))

    def test_strictly_increasing(self):
        """test some strictly_increasing examples"""
        
        assert_false(strictly_increasing(self.two_steps['x']))
        assert_false(strictly_increasing(self.two_steps_reverse['x']))
        ok_(strictly_increasing(self.two_ramps['x']))
        assert_false(strictly_increasing(self.two_ramps_reverse['x']))        
        assert_false(strictly_increasing(self.two_ramps_two_steps['x']))
        assert_false(strictly_increasing(self.two_ramps_two_steps_reverse['x']))                             
        assert_false(strictly_increasing(self.switch_back['x']))
        assert_false(strictly_increasing(self.switch_back_steps['x']))
                            
    def test_strictly_decreasing(self):
        """test some strictly_decreasing examples"""
        
        assert_false(strictly_decreasing(self.two_steps['x']))
        assert_false(strictly_decreasing(self.two_steps_reverse['x']))
        assert_false(strictly_decreasing(self.two_ramps['x']))
        ok_(strictly_decreasing(self.two_ramps_reverse['x']))        
        assert_false(strictly_decreasing(self.two_ramps_two_steps['x']))
        assert_false(strictly_decreasing(self.two_ramps_two_steps_reverse['x']))
        assert_false(strictly_decreasing(self.switch_back['x']))
        assert_false(strictly_decreasing(self.switch_back_steps['x']))
        
    def test_non_decreasing(self):
        """test some non_decreasing examples"""
        
        ok_(non_decreasing(self.two_steps['x']))
        assert_false(non_decreasing(self.two_steps_reverse['x']))
        ok_(non_decreasing(self.two_ramps['x']))
        assert_false(non_decreasing(self.two_ramps_reverse['x']))        
        ok_(non_decreasing(self.two_ramps_two_steps['x']))
        assert_false(non_decreasing(self.two_ramps_two_steps_reverse['x']))                             
        assert_false(non_decreasing(self.switch_back['x']))
        assert_false(non_decreasing(self.switch_back_steps['x']))
                            
    def test_non_increasing(self):
        """test some non_increasing examples"""
        
        assert_false(non_increasing(self.two_steps['x']))
        ok_(non_increasing(self.two_steps_reverse['x']))
        assert_false(non_increasing(self.two_ramps['x']))
        ok_(non_increasing(self.two_ramps_reverse['x']))        
        assert_false(non_increasing(self.two_ramps_two_steps['x']))
        ok_(non_increasing(self.two_ramps_two_steps_reverse['x']))
        assert_false(non_increasing(self.switch_back['x']))
        assert_false(non_increasing(self.switch_back_steps['x']))
        
    def test_non_increasing_and_non_decreasing_parts(self):
        """test some non_increasing_and_non_decreasing_parts examples"""
        assert_equal(non_increasing_and_non_decreasing_parts(self.two_steps['x']), [range(len(self.two_steps['x'])-1)])
        assert_equal(non_increasing_and_non_decreasing_parts(self.two_ramps_reverse['x']), [range(len(self.two_ramps_reverse['x'])-1)])
        assert_equal(non_increasing_and_non_decreasing_parts(self.switch_back['x']), [[0,1],[2],[3,4]])
        assert_equal(non_increasing_and_non_decreasing_parts(self.switch_back_steps['x']), [[0,1],[2,3],[4]])
        
        assert_equal(non_increasing_and_non_decreasing_parts(self.two_steps['x'],include_end_point=True), [range(len(self.two_steps['x']))])
        assert_equal(non_increasing_and_non_decreasing_parts(self.two_ramps_reverse['x'],include_end_point=True), [range(len(self.two_ramps_reverse['x']))])
        assert_equal(non_increasing_and_non_decreasing_parts(self.switch_back['x'],include_end_point=True), [[0,1,2],[2,3],[3,4,5]])
        assert_equal(non_increasing_and_non_decreasing_parts(self.switch_back_steps['x'],include_end_point=True), [[0,1,2],[2,3,4],[4,5]])
        
    def test_force_strictly_increasing(self):
        """test force_strictly_increasing"""
        x, y = force_strictly_increasing(self.two_ramps['x'], eps=0.01)                 
        ok_(np.all(x==np.array(self.two_ramps['x'])))
        
        assert_raises(ValueError, force_strictly_increasing, self.switch_back['x'])
        
        x, y = force_strictly_increasing(self.two_ramps_two_steps['x'], self.two_ramps_two_steps['y'], keep_end_points = True, eps=0.01)                 
        ok_(np.allclose(x, np.array([0,  0.38,   0.4,  1,  2.5,  2.99,  3])))
        ok_(np.allclose(y, np.array(self.two_ramps_two_steps['y'])))
        
        x, y = force_strictly_increasing(self.two_ramps_two_steps['x'], self.two_ramps_two_steps['y'], keep_end_points = False, eps=0.01)                 
        ok_(np.allclose(x, np.array([0,  0.4,   0.41,  1,  2.5,  3,  3.02])))

        x, y = force_strictly_increasing(self.two_ramps_two_steps_reverse['x'], self.two_ramps_two_steps_reverse['y'], keep_end_points = False, eps=0.01)                 
        ok_(np.allclose(x, np.array([-3, -2.99, -2.5, -1, -0.4, -0.38, 0])))
        ok_(np.allclose(y, np.array(self.two_ramps_two_steps['y'][::-1])))
    
    def test_force_non_decreasing(self):
        """test force_non_decreasing"""
        x, y = force_non_decreasing(self.two_ramps_two_steps['x'], self.two_ramps_two_steps['y'])                 
        ok_(np.all(x==np.array(self.two_ramps_two_steps['x'])))
        ok_(np.all(y==np.array(self.two_ramps_two_steps['y'])))
        
        assert_raises(ValueError, force_non_decreasing, self.switch_back['x'])
        
        x, y = force_non_decreasing(self.two_ramps_two_steps_reverse['x'], self.two_ramps_two_steps_reverse['y'])                 
        ok_(np.all(x==np.array([-3, -3, -2.5, -1, -0.4, -0.4, 0])))
        ok_(np.all(y==np.array(self.two_ramps_two_steps['y'][::-1])))        
        
        
    def test_ramps_constants_steps(self):
        ramps, constants, steps = ramps_constants_steps(self.two_steps['x'], self.two_steps['y'])
        ok_(np.all(ramps==np.array([])))
        ok_(np.all(constants==np.array([1,3])))        
        ok_(np.all(steps==np.array([0,2])))
        
        ramps, constants, steps = ramps_constants_steps(self.two_ramps['x'], self.two_ramps['y'])
        ok_(np.all(ramps==np.array([0,2])))
        ok_(np.all(constants==np.array([1,3])))        
        ok_(np.all(steps==np.array([])))
        
        ramps, constants, steps = ramps_constants_steps(self.two_ramps_two_steps['x'], self.two_ramps_two_steps['y'])
        ok_(np.all(ramps==np.array([0,3])))
        ok_(np.all(constants==np.array([2,4])))        
        ok_(np.all(steps==np.array([1,5])))
        
#               0      1      2     3     4     5    6
#        {'x': [0,  -0.4,   -0.4,  -1,  -2.5,  -3,  -3],
#         'y': [0,  10.0,   20.0,  20,  30.0, 30, 40]}

    def test_start_index_of_steps(self):
        ok_(np.allclose(start_index_of_steps(**self.two_steps),np.array([0,2])))
        ok_(np.allclose(start_index_of_steps(**self.two_ramps),np.array([])))
        ok_(np.allclose(start_index_of_steps(**self.two_ramps_two_steps),np.array([1,5])))

    def test_start_index_of_ramps(self):
        ok_(np.allclose(start_index_of_ramps(**self.two_steps),np.array([])))
        ok_(np.allclose(start_index_of_ramps(**self.two_ramps),np.array([0,2])))
        ok_(np.allclose(start_index_of_ramps(**self.two_ramps_two_steps),np.array([0,3])))
        
    def test_start_index_of_constants(self):
        ok_(np.allclose(start_index_of_constants(**self.two_steps),np.array([1,3])))
        ok_(np.allclose(start_index_of_constants(**self.two_ramps),np.array([1,3])))
        ok_(np.allclose(start_index_of_constants(**self.two_ramps_two_steps),np.array([2,4])))        
        
#class test_has_steps(sample_linear_piecewise):
#    """test some has_steps examples"""
#    def __init__(self):
#        sample_linear_piecewise.__init__(self)                                            
#        
#    def test_(self):
#        """test some has_steps examples"""
#        
#        assert_false(has_steps(self.two_steps['x'], self.two_steps['y']))
#        ok_(has_steps(self.two_steps_reverse['x'], self.two_steps_reverse['y']))
#        assert_false(has_steps(self.two_ramps['x'], self.two_ramps['y']))
#        assert_false(has_steps(self.two_ramps_reverse['x'], self.two_ramps_reverse['y']))        
#        ok_(has_steps(self.two_ramps_two_steps['x'], self.two_ramps_two_steps['y']))
#        ok_(has_steps(self.two_ramps_two_steps_reverse['x'], self.two_ramps_two_steps_reverse['y']))        