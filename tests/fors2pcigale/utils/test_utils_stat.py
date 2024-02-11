"""Module to test utils module"""

# pylint: disable=line-too-long
# pylint: disable=missing-final-newline
# pylint: disable=W0612

# python -m unittest test_utils_stat.py

import unittest

import numpy as np

from fors2pcigale.utils.utils_stat import weighted_mean, weighted_variance


class UtilsStatTestCase(unittest.TestCase):
    """A test case for the utils package."""

    def test_weighted_mean(self):
        """test if weighted_mean works
        """
        
        self.assertAlmostEqual(weighted_mean(np.array([1.,1.])),1.)
        self.assertAlmostEqual(weighted_mean(np.array([1.,1.,1,1.])),1.)

    def weighted_variance(self):
        """test if weighted_variance works
        """
        
        self.assertAlmostEqual(weighted_variance(np.array([1.,1.]),1./np.array([1.,1.])),1/np.sqrt(2.))
        self.assertAlmostEqual(weighted_variance(np.array([1.,1.,1,1.]),1./np.array([1.,1.,1.,1.]) ),1./2.)

        self.assertAlmostEqual(weighted_variance(np.array([1.,1.])),1/np.sqrt(2.))
        self.assertAlmostEqual(weighted_variance(np.array([1.,1.,1,1.])),1./2.)


if __name__ == "__main__":
    unittest.main()