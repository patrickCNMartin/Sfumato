import unittest
import numpy as np

import sys
sys.path.append('../src')

from preprocessing import *
from unittest.mock import patch


###############################################################################

class TestCorrelatedRemoval(unittest.TestCase):

    # --------------------------------------- #
    #  tests for the function getting_bags    #
    # --------------------------------------- #

    def test_getting_bags_EMPTY(self):
        edges = []
        solution = []
        self.assertEqual(getting_bags(edges), solution)
    

    def test_getting_bags_1_EDGE(self):
        edges = [(0,1)]
        solution = [{0,1}]
        self.assertEqual(getting_bags(edges), solution)
    

    def test_getting_bags_2_SEQ_EDGES(self):
        edges = [(0,1), (1,2)]
        solution = [{0,1,2}]
        self.assertEqual(getting_bags(edges), solution)
    

    def test_getting_bags_3_NEST_EDGES(self):
        edges = [(0,1), (0,2), (2,3)]
        solution = [{0,1,2,3}]
        self.assertEqual(getting_bags(edges), solution)
    

    def test_getting_bags_2_BAGS_1_EDGE(self):
        edges = [(0,1), (2,3)]
        solution = [{0,1}, {2,3}]
        self.assertEqual(getting_bags(edges), solution)
    

    def test_getting_bags_2_BAGS_MULT_EDGES(self):
        edges = [(0,1), (1,2), (3,4), (4,5)]
        solution = [{0,1,2}, {3,4,5}]
        self.assertEqual(getting_bags(edges), solution)
    

    def test_getting_bags_2_BAGS_NEST_EDGES(self):
        edges = [(0,1), (1,2), (1,3), (1,4), (3,4), (4,5), (7,8), (7,9), (8,12)]
        solution = [{0,1,2,3,4,5}, {7,8,9,12}]
        self.assertEqual(getting_bags(edges), solution)
    

    # -------------------------------------------------- #
    #   tests for the function removing_cols_from_bags   #
    # -------------------------------------------------- #

    def test_removing_cols_from_bags_EMPTY(self):
        bags = []
        solution = set()
        self.assertEqual(removing_cols_from_bags(bags), solution)


    @patch('preprocessing.choice', lambda x: 0)
    def test_removing_cols_from_bags_2_ELEM_1_BAG_0(self):
        bags = [{0,1}]
        solution = {1}
        self.assertEqual(removing_cols_from_bags(bags), solution)
    

    @patch('preprocessing.choice', lambda x: 1)
    def test_removing_cols_from_bags_2_ELEM_1_BAG_1(self):
        bags = [{0,1}]
        solution = {0}
        self.assertEqual(removing_cols_from_bags(bags), solution)
    

    @patch('preprocessing.choice', lambda x: 5)
    def test_removing_cols_from_bags_4_ELEM_1_BAG(self):
        bags = [{0,1,5,6}]
        solution = {0,1,6}
        self.assertEqual(removing_cols_from_bags(bags), solution)
    

    @patch('preprocessing.choice', lambda x: 3)
    def test_removing_cols_from_bags_MULT_ELEM_2_BAG(self):
        bags = [{0,1,2,4,5,3}, {10,12,14,3,140}]
        solution = {0,1,2,4,5,10,12,14,140}
        self.assertEqual(removing_cols_from_bags(bags), solution)
    

    @patch('preprocessing.choice', lambda x: 10)
    def test_removing_cols_from_bags_MULT_ELEM_MULT_BAGS(self):
        bags = [{0,4,10}, {7, 40, 10}, {6,2,10}, {8,33,10}]
        solution = {0,2,4,6,7,8,33,40}
        self.assertEqual(removing_cols_from_bags(bags), solution)


    # --------------------------------------- #
    #  tests for the function find_corr_cols  #
    # --------------------------------------- #

    def test_apx_vertex_cover_NO_EDGES(self):
        edges = []
        solution = set()
        self.assertEqual(apx_vertex_cover(edges), solution)


    def test_apx_vertex_cover_1_EDGE(self):
        edges = [(0,1)]
        solution = set()
        self.assertEqual(apx_vertex_cover(edges), solution)


    def test_apx_vertex_cover_2_INTERSECT_EDGE_0_COM(self):
        edges = [(0,1), (0,2)]
        solution = {2}
        self.assertEqual(apx_vertex_cover(edges), solution)


    def test_apx_vertex_cover_2_INTERSECT_EDGE_1_COM(self):
        edges = [(0,1), (1,2)]
        solution = {2}
        self.assertEqual(apx_vertex_cover(edges), solution)


    def test_apx_vertex_cover_2_INTERSECT_EDGE_1_COM(self):
        edges = [(0,1), (1,2)]
        solution = {2}
        self.assertEqual(apx_vertex_cover(edges), solution)


    def test_apx_vertex_cover_2_NO_INTERSECT_EDGE(self):
        edges = [(0,1), (4,5)]
        solution = set()
        self.assertEqual(apx_vertex_cover(edges), solution)


    def test_apx_vertex_cover_SEQ_INTERSECT_EDGE(self):
        edges = [(0,1), (1,2), (2,3)]
        solution = set()
        self.assertEqual(apx_vertex_cover(edges), solution)


    def test_apx_vertex_cover_MULT_INTERSECT_EDGE(self):
        edges = [(0,1), (0,2), (1,3), (1,4), (2,5), (6,7), (6,10), (7, 13)]
        solution = {3, 4, 10, 13}
        self.assertEqual(apx_vertex_cover(edges), solution)


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
    
    filename = "../../data/Puck_190926_06_combined.csv"

    cm, bc_met, gene_met = load_and_preprocess(filename, 
                                bc_min={'counted_genes': 10, 'variance': 0.001},
                                gene_top={'variance': 10})
    print(f"Size of the matrix {cm.shape}")
    print()
    print(bc_met.head(6))
    print()
    print(gene_met.head(6))
    print()


    ################################
    time_elapsed = (time.perf_counter() - time_start)
    memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    print ("%5.1f secs %5.1f MByte" % (time_elapsed,memMb))
    print()
    
    print("")

    # unittest.main()