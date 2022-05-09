import csv
from itertools import count

import numpy as np
import pandas as pd
import multiprocessing as mp

from typing import Generator

###############################################################################
# loading data
###############################################################################

# ---------------------------- #
# readers and type converters  #
# ---------------------------- #

def readr_generator(filename: str, delimiter: str=",") -> Generator:
    """Reads a file line by line, while converting each row into a list, by 
    separating each of the values by the delimiter.

    Args:
        filename (str): name and path of the file that contains the data.
        delimiter (str, optional): delimiter of the fields of the datafile. 
            Defaults to ",".

    Yields:
        Generator: object that allows for the iteration of the lines of the 
            datafile. Each line is represented as a list of the fields separated
            by the delimiter.
    """

    with open(filename, 'r', encoding='utf8') as datafile:
        data_reader = csv.reader(datafile, delimiter=delimiter)
        for i, line in enumerate(data_reader):
            yield line, i


def sep_tag_counts(row: list, index: int, skip_fst_col: bool=True) -> tuple:
    """Given a row of data, it separates the tags from the counts.

    Args:
        row (list): Row of data with the following format: barcode, 
            x coordinate, y coordinate, gene 1, gene ..., gene n.
        index (int): integer to serve as a key to connect the tags to the 
            counts.
        skip_fst_col (bool, optional): flag indicate whether to consider the 
            first column or not. Defaults to True.

    Returns:
        tags (dict): tags of the row (barcode, x coordinate, y coordinate, 
            index).
        (np.ndarray): counts of the genes and the index.
    """
    
    tags = {'barcode': row[skip_fst_col], 
            'x_coor': float(row[skip_fst_col + 1]), 
            'y_coor': float(row[skip_fst_col + 2]),
            'index': index}
    
    return (tags, np.array(row[skip_fst_col+3:] + [index], dtype=int))


# ---------------------------- #
#       barcode metrics        #
# ---------------------------- #

def row_n_genes(tags: dict, gene_counts: np.ndarray):
    """Calculates the number of different genes that were counted at least once,
    in a given barcode.

    Args:
        tags (dict): tags of the row (barcode, x coordinate, y coordinate, 
            index).
        gene_counts (np.ndarray): counts of each gene for the current barcode.
    """

    tags['counted_genes'] = np.sum(gene_counts > 0)


def row_gene_counts(tags: dict, gene_counts: np.ndarray):
    """Calculates the total number of genes that were counted in a given 
    barcode.

    Args:
        tags (dict): tags of the row (barcode, x coordinate, y coordinate, 
            index).
        gene_counts (np.ndarray): counts of each gene for the current barcode.
    """

    tags['total_counts'] = np.sum(gene_counts)


def row_gene_var(tags: dict, gene_counts: np.ndarray):
    """Calculates the variance of the gene count in a given barcode.

    Args:
        tags (dict): tags of the row (barcode, x coordinate, y coordinate, 
            index).
        gene_counts (np.ndarray): counts of the genes for a given barcode.
    """

    tags['variance'] = np.var(gene_counts)


def calc_row_metrics(tags: dict, gene_counts: np.ndarray) -> dict:
    """Calculates the row wise metrics.

    Args:
        tags (dict): tags of the row (barcode, x coordinate, y coordinate, 
            index).
        gene_counts (np.ndarray): counts of the genes for a given barcode.

    Returns:
        dict: tags with the calculated metrics. Namely, the number of different
            genes counted at least once, the total gene counts and the variance
            in the gene counts, for a given barcode.
    """
    
    row_n_genes(tags, gene_counts)
    row_gene_counts(tags, gene_counts)
    row_gene_var(tags, gene_counts)
    return tags

###############################################################################

if __name__ == '__main__':
    import time
    import resource
    import functools

    time_start = time.perf_counter()

    ################################
    # actual code
    ################################
    
    d = readr_generator("../../data/Puck_190926_06_combined.csv")
    next(d)
    def z(t):
        tags, gene_counts = sep_tag_counts(*t)
        return calc_row_metrics(tags, gene_counts)
    
    h = functools.reduce(lambda acc, j: acc + [z(j)], d, [])
    print(pd.DataFrame(h).head(10))

    ################################
    time_elapsed = (time.perf_counter() - time_start)
    memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    print ("%5.1f secs %5.1f MByte" % (time_elapsed,memMb))