import unittest
import numpy as np

import sys
sys.path.append('../src')

from preprocessing import *


###############################################################################

class TestCorrelatedRemoval(unittest.TestCase):

    # --------------------------------------- #
    #  tests for the function find_corr_cols  #
    # --------------------------------------- #

    def test_find_corr_cols_NO_CORR(self):
        corr_matrix = np.zeros((6,6)) == 1
        solution = []
        self.assertEqual(find_corr_cols(corr_matrix), solution)


    def test_find_corr_cols_1_CORR(self):
        corr_matrix = np.ones((2,2)) == 1
        solution = [(0,1)]
        self.assertEqual(find_corr_cols(corr_matrix), solution)


    def test_find_corr_cols_1_MULT_CORR(self):
        corr_matrix = np.array([[1,1,1], 
                                [1,1,0], 
                                [1,0,1]])
        corr_matrix = corr_matrix == 1
        solution = [(0,1), (0,2)]
        self.assertEqual(find_corr_cols(corr_matrix), solution)


    def test_find_corr_cols_1_1_CORR(self):
        corr_matrix = np.array([[1,1,0], 
                                [1,1,1], 
                                [0,1,1]]) 
        corr_matrix = corr_matrix == 1
        solution = [(0,1), (1,2)]
        self.assertEqual(find_corr_cols(corr_matrix), solution)


    def test_find_corr_cols_1_NEST_CORR(self):
        corr_matrix = np.ones((3,3)) == 1
        solution = [(0,1), (0,2), (1,2)]
        self.assertEqual(find_corr_cols(corr_matrix), solution)


    def test_find_corr_cols_1_NEST_EMPTY_CORR(self):
        corr_matrix = np.array([[1,1,1,0], 
                                [1,1,1,0], 
                                [1,1,1,0], 
                                [0,0,0,1]])
        corr_matrix = corr_matrix == 1
        solution = [(0,1), (0,2), (1,2)]
        self.assertEqual(find_corr_cols(corr_matrix), solution)


    def test_find_corr_cols_1_NEST_SINGLE_CORR(self):
        corr_matrix = np.array([[1,1,1,0], 
                                [1,1,1,1], 
                                [1,1,1,0], 
                                [0,1,0,1]])
        corr_matrix = corr_matrix == 1
        solution = [(0,1), (0,2), (1,2), (1,3)]
        self.assertEqual(find_corr_cols(corr_matrix), solution)


    def test_find_corr_cols_1_NEST_FULL_CORR(self):
        corr_matrix = np.ones((4,4)) == 1
        solution = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        self.assertEqual(find_corr_cols(corr_matrix), solution)


    def test_find_corr_cols_2_SINGLE_CORR(self):
        corr_matrix = np.array([[1,1,0,0,0], 
                                [1,1,0,0,0], 
                                [0,0,1,0,0],
                                [0,0,0,1,1], 
                                [0,0,0,1,1]])
        bool_corr_matrix = corr_matrix == 1
        solution = [(0,1), (3,4)]
        self.assertEqual(find_corr_cols(bool_corr_matrix), solution)



###############################################################################

if __name__ == '__main__':
    import time
    import resource

    time_start = time.perf_counter()

    ################################
    # actual code
    ################################
    
    g = loader("../../data/Puck_190926_06_combined.csv")[1]
    cm, bc_met = filter_barcodes(g, bc_min={'counted_genes': 5, 
                                            'variance': 0.01},
                                    bc_max={'variance':4},
                                    bc_top={'total_counts': 50})
    print(bc_met.head(10))
    print(f"Size of the matrix {cm.shape}")
    print()


    ################################
    time_elapsed = (time.perf_counter() - time_start)
    memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    print ("%5.1f secs %5.1f MByte" % (time_elapsed,memMb))
    print()
    
    print("")

    unittest.main()