import csv

import numpy as np
import numba as nb
import pandas as pd

from math import ceil
from typing import Generator, Iterable
from random import choice
from sklearn import preprocessing
from functools import reduce, partial
from collections import deque
from scipy.stats import spearmanr


###############################################################################
# implementing robustness

def is_chosen_available(available: dict or set, chosen: str, metric: bool):
    """Checks if a given metric or method is part of the available 
    metrics/methods. In case it is not, it raises an error.

    Args:
        available (dict or set): metrics or methods that are available to be 
            used.
        chosen (str): metric or method chosen to be used.
        metric (bool): flag indicating if the value chosen is a metric or a 
            method. If it is true, it indicates that the value chosen is a 
            metric. If it is false, it indicates that the value chosen is a 
            method.

    Raises:
        ValueError: error raised when a metric or method does not belong to the 
            set of possible metrics/methods.
    """
    if chosen in available:
        return None

    possible = available if available is set else available.keys()
    name = "metric" if metric else "method"

    raise ValueError(f"You chose the following {name}: '{chosen}'." +
                        f"This {name} is not available. The {name} chosen " +
                        "must correspond to one of the following " +
                        f"{name}s: {possible}")


def check_multiple_metrics(available_metrics: dict, chosen_metrics: set):
    """Checks if the chosen metrics are among the available metrics.

    Args:
        available_metrics (dict): available metrics to filter the data.
        chosen_metrics (set): metrics chosen to filter the barcodes by.

    Raises:
        ValueError: error raised when a metric does not belong to the set of 
            possible metrics.
    """
    
    for metric in chosen_metrics:
        is_chosen_available(available_metrics, metric, True)


###############################################################################
# loading data

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
        for line in data_reader:
            yield line


def obtain_gene_names(readr: Generator, skip_fst_col: bool=True) -> tuple:
    """Obtains the gene names.

    Args:
        readr (Generator): generator of the rows of the data.

    Returns:
        headers (list[str]): names of the genes.
        readr (Generator): generator of the rows of the data without the first
            row.
    """

    headers = next(readr)[skip_fst_col+3:]
    return headers, readr


def sep_tag_counts(row: list, skip_fst_col: bool=True) -> tuple:
    """Given a row of data, it separates the tags from the counts.

    Args:
        row (list): Row of data with the following format: barcode, 
            x coordinate, y coordinate, gene 1, gene ..., gene n.
        skip_fst_col (bool, optional): flag indicate whether to consider the 
            first column or not. Defaults to True.

    Returns:
        tags (dict): tags of the row (barcode, x coordinate, y coordinate).
        (np.ndarray): counts of the genes and the index.
    """
    
    tags = {'barcode': row[skip_fst_col], 
            'x_coor': float(row[skip_fst_col + 1]), 
            'y_coor': float(row[skip_fst_col + 2])}
    
    return (tags, np.array(row[skip_fst_col+3:], dtype=int))


# ---------------------------- #
#         data loader          #
# ---------------------------- #


def loader(filename: str, delimiter: str=",", skip_fst_col: bool=True) -> tuple:
    """Loads the data, extracts the column names and separates the 
    barcode column, the x coordinate column and the y coordinate column from the
    gene counts columns.

    Args:
        filename (str): name and path of the file.
        delimiter (str, optional): field/. Defaults to ",".
        skip_fst_col (bool, optional): _description_. Defaults to True.

    Returns:
        gene_names (list[str]): names of the genes counted in the data.
        row_gen (Generator): generator of the rows of the data.
    
    Requires:
        filename: should be a valid path/name of the file.

    Ensures:
        row_gen: each row generated by row_gen is a 2 element tuple with the 
            first element being a dictionary of the barcode, x coordinate, y 
            coordinate and their associated values, and the second element is
            the counts of the genes associated with that barcode.
    """

    data = readr_generator(filename, delimiter)
    gene_names, data = obtain_gene_names(data)

    row_gen = map(lambda row: sep_tag_counts(row, skip_fst_col), data)
    
    return gene_names, row_gen


###############################################################################
# metrics

@nb.njit
def addition(vector: np.ndarray) -> int:
    """Wrapper for the numpy sum function.

    Args:
        vector (np.ndarray): vector of numerical values.

    Returns:
        int: sum of the values of the given vector.
    """

    return np.sum(vector)


@nb.njit
def variance(vector: np.ndarray) -> float:
    """Wrapper for the numpy variance function.

    Args:
        vector (np.ndarray): vector of numerical values.

    Returns:
        float: variance of the given vector.
    """

    return np.var(vector)


@nb.njit
def dispersion_ratio(vector: np.ndarray) -> float:
    """Calculates the dispersion ratio (ratio of the arithmetic mean and the 
    geometric mean).

    Args:
        vector (np.ndarray): vector of numerical values.

    Returns:
        float: ratio of arithmetic mean over geometric mean.
    """

    vector += 1
    gm = 10**((1/vector.size)*np.sum(np.log(vector)))
    return np.mean(vector)/gm


@nb.njit
def mad(vector: np.ndarray) -> float:
    """Calculates the mean absolute difference.

    Args:
        vector (np.ndarray): vector of numerical values.

    Returns:
        float: mean absolute difference.
    """

    return np.sum(np.abs(vector - np.mean(vector)))/vector.size


###############################################################################
# barcode filtering / row filtering

# ------------------------------- #
#  calculating barcode metrics    #
# ------------------------------- #

def row_n_genes(tags: dict, gene_counts: np.ndarray):
    """Calculates the number of different genes that were counted at least once,
    in a given barcode.

    Args:
        tags (dict): tags of the row (barcode, x coordinate, y coordinate).
        gene_counts (np.ndarray): counts of each gene for the current barcode.
    """

    tags['counted_genes'] = addition(gene_counts > 0)


def row_gene_counts(tags: dict, gene_counts: np.ndarray):
    """Calculates the total number of genes that were counted in a given 
    barcode.

    Args:
        tags (dict): tags of the row (barcode, x coordinate, y coordinate).
        gene_counts (np.ndarray): counts of each gene for the current barcode.
    """

    tags['total_counts'] = addition(gene_counts)


def row_gene_var(tags: dict, gene_counts: np.ndarray):
    """Calculates the variance of the gene count in a given barcode.

    Args:
        tags (dict): tags of the row (barcode, x coordinate, y coordinate).
        gene_counts (np.ndarray): counts of the genes for a given barcode.
    """

    tags['variance'] = variance(gene_counts)


def row_gene_mad(tags: dict, gene_counts: np.ndarray):
    """Calculates the mean absolute difference of the gene count in a given 
    barcode.

    Args:
        tags (dict): tags of the row (barcode, x coordinate, y coordinate).
        gene_counts (np.ndarray): counts of the genes for a given barcode.
    """

    tags['mad'] = mad(gene_counts)


def row_gene_dispersion(tags: dict, gene_counts: np.ndarray):
    """Calculates the ratio of the arithmetic mean to geometric mean of the gene
    count.

    Args:
        tags (dict): tags of the row (barcode, x coordinate, y coordinate).
        gene_counts (np.ndarray): counts of the genes for a given barcode.
    """

    tags['dispersion'] = dispersion_ratio(gene_counts)


def calc_row_metrics(tags: dict, gene_counts: np.ndarray, metrics: set, 
                        metric_funcs: dict) -> dict:
    """Calculates the given metrics for a barcode.

    Args:
        tags (dict): barcode, x coordinate and y coordinate of each row of the
            original data.
        gene_counts (np.ndarray): gene counts associated with the barcode of the
            tags.
        metrics (set): metrics that are wanted to be calculated using the gene
            counts of a given barcode.
        metric_funcs (dict): metric names associated with the function that 
            performs the calculation of the metric.

    Returns:
        tags (dict): barcode, x coordinate, y coordinate and metrics related to
            the barcode.
    """
    
    for metric in metrics:
        metric_funcs[metric](tags, gene_counts)
    
    return tags


# -------------------------------------- #
# filtering barcodes on absolute values  #
# -------------------------------------- #

def filter_abs_barcodes(tags: dict, min: dict, max: dict) -> bool:
    """Filters barcodes based on absolute values of the defined metrics.

    Args:
        tags (dict): barcode, x coordinate, y coordinate and assossiated 
            metrics.
        min (dict): metrics (key) for which a minimum value (value) is required 
            for the barcode to be included in the downstream analysis.
        max (dict): metrics (key) for which a maximum value (value) is required 
            for the barcode to be included in the downstream analysis.
    
    Returns:
        bool: flag indicating if the barcode should be included in the 
            downstream analysis (True) or not (False).
    """
    
    for metric in min:
        if tags[metric] < min[metric]:
            return False
        
    for metric in max:
        if tags[metric] > max[metric]:
            return False
    
    return True


# -------------------------------------- #
# filtering barcodes on relative values  #
# -------------------------------------- #

def subset_df(df: pd.DataFrame, top: float, bottom: float) -> pd.DataFrame:
    """Subsets a sorted dataframe into its top rows and bottom rows if they
    are defined.

    Args:
        df (pd.DataFrame): dataframe to subset
        top (float): number of rows with the highest values to include in the
            subset.
        bottom (float): number of rows with the lowest values to include in the
            subset.
    
    Returns:
        pd.DataFrame: subset of the original dataframe with the defined rows for
            the highest and lowest values.
    """
    
    if top and bottom:
        retain_top = ceil(top/100*df.shape[0])
        retain_bottom = df.shape[0] - ceil(bottom/100*df.shape[0]) -1
        df.drop(np.arange(retain_top, retain_bottom), axis=0, inplace=True)
        return df
    
    if top:
        n_retain = ceil(top/100*df.shape[0])
    else:
        n_retain = ceil(bottom/100*df.shape[0])  
    
    return df.head(n_retain)


def filter_rel_bc_on_metric(count_matrix: np.ndarray, 
                            barcode_metrics: pd.DataFrame, metric: str,
                            top: float=None, bottom: float=None) -> tuple:
    """Filters the count_matrix and the barcode metrics based on a column of the
    barcode metrics. It obtains either the rows associated with the highest 
    values of the defined metric, the lowest values or both.

    Args:
        count_matrix (np.ndarray): matrix of the counts of measured genes in 
            each barcode (barcode x genes).
        barcode_metrics (pd.DataFrame): barcode metrics (barcode, xcoord, 
            ycoord, counted_genes, total_counts, variance).
        metric (str): column of the barcode metrics used to subset the 
            data.
        top (float, optional): if defined, corresponds to the number of rows 
            associated with the highest values of the metric wanted in 
            percentage. Defaults to None.
        bottom (float, optional):  if defined, corresponds to the number of rows 
            associated with the lowest values of the metric wanted in 
            percentage. Defaults to None.

    Returns:
        count_matrix (np.ndarray): filtered matrix of the counts of measured 
            genes in each barcode (barcode x genes).
        barcode_metrics (pd.DataFrame): filtered dataframe of barcode metrics.
    """

    barcode_metrics.sort_values(metric, ascending=top == None, inplace=True)
    barcode_metrics = subset_df(barcode_metrics, top, bottom)
    
    count_matrix = count_matrix[barcode_metrics.index, :]
    
    barcode_metrics.reset_index(inplace = True)
    barcode_metrics = barcode_metrics.drop('index', axis=1) 

    return count_matrix, barcode_metrics


def filter_rel_mult_met(cm: np.ndarray, met_df: pd.DataFrame, top_met: dict, 
                            bottom_met: dict, rel_filt_func: callable) -> tuple:
    """Performs filtering based on the relative value of each metric given in 
    bc_bottom and/or bc_top of the barcode.

    Args:
        cm (np.ndarray): count matrix of the genes for each barcode. Matrix 
            (barcode x genes).
        met_df (pd.DataFrame): dataframe metrics.
        top_met (dict): percentage of elements with the highest value of each 
            metric that are to be used in the downstream analysis. 
        bottom_met (dict): percentage of elements with the lowest value of each 
            metric that are to be used in the downstream analysis.
        rel_filt_func (callable): function that is going to filter the count 
            matrix and the dataframe of metrics based on the relative values 
            given.

    Returns:
        cm (np.ndarray): filtered count matrix with only the columns/rows
            that fall in a given range from the highest values and/or lowest 
            values of each specified metric.
        met_df (pd.DataFrame): filtered metrics dataframe with only the 
            columns/rows that fall in a given range from the highest values 
            and/or lowest values of each specified metric.
    """

    for metric in top_met.keys() | bottom_met.keys():
        top = top_met[metric] if metric in top_met else None
        bottom = bottom_met[metric] if metric in bottom_met else None
        cm, met_df = rel_filt_func(cm, met_df, metric, top, bottom)
    
    return cm, met_df


# ---------------------------- #
#      barcode filtering       #
# ---------------------------- #


def filter_barcodes(barcodes: Generator, bc_min: dict={}, bc_max: dict={}, 
                    bc_top: dict={}, bc_bottom: dict={}) -> tuple:
    """Constructs a dataframe of the barcode, position and associated metrics,
    while filtering said dataframe and the count matrix of associated genes by
    absolute and relative values of the barcode metrics.

    Args:
        barcodes (Generator): generator of rows/barcodes.
        bc_min (dict, optional):  metrics (key) for which a minimum value 
            (value) is required for each barcode to be included in the 
            downstream analysis.
        bc_max (dict, optional): metrics (key) for which a maximum value (value)
            is required for each barcode to be included in the downstream 
            analysis. Defaults to {}.
        bc_top (dict, optional): percentage of barcodes with the highest value 
            of each metric that are to be used in the downstream analysis.
            Defaults to {}.
        bc_bottom (dict, optional): percentage of barcodes with the lowest value
            of each metric that are to be used in the downstream analysis.
            Defaults to {}.

    Returns:
        count_matrix (np.ndarray): filtered count matrix with only the barcodes
            that fall in a given range from the highest values and/or lowest 
            values of each specified metric and with the values within a 
            specified range of values specified by the given metrics.
        barcode_metrics (pd.DataFrame): filtered barcode metrics with only the 
            barcodes that fall in a given range from the highest values and/or 
            lowest values of each specified metric and with the values within a 
            specified range of values specified by the given metrics.
    
    Requires:
        barcodes: the generator must generate 2 element tuples with the first 
            element being the barcode, x coordinate and y coordinate, and the 
            second element being the gene counts of the barcode.
    """
    
    metrics = {'counted_genes': row_n_genes, 'total_counts': row_gene_counts,
                'variance': row_gene_var, 'mad': row_gene_mad, 
                'dispersion': row_gene_dispersion}
    
    chosen_metrics = (bc_min.keys() | bc_max.keys() | bc_bottom.keys() |
                        bc_top.keys())
    
    check_multiple_metrics(metrics, chosen_metrics)
    
    rw_met_f = lambda row: (calc_row_metrics(row[0], row[1], chosen_metrics, 
                                                metrics), row[1])
    filtered_bc = filter(lambda bc: filter_abs_barcodes(bc[0], bc_min, bc_max), 
                            map(rw_met_f, barcodes))

    joiner = lambda acc, bc: (acc[0] + [bc[0]], acc[1] + [bc[1]])
    barcode_met, count_matrix = reduce(joiner, filtered_bc, ([], []))

    count_matrix = np.array(count_matrix)
    barcode_met = pd.DataFrame(barcode_met)
    return filter_rel_mult_met(count_matrix, barcode_met, bc_top, bc_bottom,
                                filter_rel_bc_on_metric)


###############################################################################
# gene filtering / columnn filtering

# ------------------------------- #
#     calculating gene metrics    #
# ------------------------------- #

def calc_barcode_per_gene(gene_counts: np.ndarray) -> np.ndarray:
    """Calculates the number of barcodes that contain each gene.

    Args:
        gene_counts (np.ndarray): count matrix of barcodes x genes.

    Returns:
        np.ndarray: 1-D array of the number of barcodes that measured that gene.
    """

    return np.sum(gene_counts > 0, axis=0)


def calc_total_measures_per_gene(gene_counts: np.ndarray) -> np.ndarray:
    """Calculates the sum of the number of times each gene was measured accross
    all barcodes.

    Args:
        gene_counts (np.ndarray): count matrix of barcodes x genes.

    Returns:
        np.ndarray: 1-D array of the number of times each gene was measured.
    """

    return np.sum(gene_counts, axis=0)


def calc_var_per_gene(gene_counts: np.ndarray) -> np.ndarray:
    """Calculates the variance of the number of measures of each gene.

    Args:
        gene_counts (np.ndarray): count matrix of barcodes x genes.

    Returns:
        np.ndarray: 1-D array of the variance of the measure count of each gene.
    """

    return np.var(gene_counts, axis=0)


def calc_dispersion_per_gene(gene_counts: np.ndarray) -> np.ndarray:
    """Calculates the dispersion ratio (ratio of the arithmetic mean to the 
    geometric mean) of the measures of each gene.

    Args:
        gene_counts (np.ndarray): count matrix of barcodes x genes.

    Returns:
        np.ndarray: 1-D array of the dispersion of the measure count of each 
            gene.
    """

    return np.apply_along_axis(dispersion_ratio, 0, gene_counts)


def calc_mad_per_gene(gene_counts: np.ndarray) -> np.ndarray:
    """Calculates the mean average deviation (MAD) of the measures of each gene.

    Args:
        gene_counts (np.ndarray): count matrix of barcodes x genes.

    Returns:
        np.ndarray: 1-D array of the MAD of the measure count of each gene.
    """

    return np.apply_along_axis(mad, 0, gene_counts)


def calc_gene_metrics(gene_names: list, gene_counts: np.ndarray, 
                        metrics: Iterable, metric_funcs: dict) -> pd.DataFrame:
    """Constructs a dataframe with the specified metrics regarding each gene.

    Args:
        gene_names (list): names of the genes in the count matrix.
        gene_counts (np.ndarray): count matrix of barcodes x genes.
        metrics (Iterable[str]): names of the metrics to be calculated.
        metric_funcs (dict[str] -> Callable): names of the metrics associated 
            with the functions that calculate those metrics for each gene.

    Returns:
        genes (pd.DataFrame): specified metrics regarding each gene. 
    """
    
    genes = pd.DataFrame({'genes': gene_names})
    
    for metric in metrics:
        genes[metric] = metric_funcs[metric](gene_counts)

    return genes


# ----------------------------------- #
# filtering genes on absolute values  #
# ----------------------------------- #

def filter_abs_genes(count_matrix: np.ndarray, gene_metrics: pd.DataFrame, 
                        gene_min: dict, gene_max: dict) -> tuple:
    """Filters the genes based on the pure value of the metrics (as posed to 
    based on the comparison of values).

    Args:
        count_matrix (np.ndarray): matrix of the counts of measured genes in 
            each barcode (barcode x genes).
        gene_metrics (pd.DataFrame): gene metrics (genes, n_barcodes, 
            total_measures, variance).
        gene_min (dict): metrics (key) for which a minimum value (value) is 
            required for a gene to be included in the downstream analysis.
        gene_max (dict): metrics (key) for which a maximum value (value) is 
            required for a gene to be included in the downstream analysis.

    Returns:
        count_matrix (np.ndarray): filtered matrix of the counts of measured 
            genes in each barcode (barcode x genes).
        gene_metrics (pd.DataFrame): filtered dataframe of gene metrics.
    """
    
    for metric in gene_min:
        gene_metrics = gene_metrics[gene_metrics[metric] >= gene_min[metric]]
    
    for metric in gene_max:
        gene_metrics = gene_metrics[gene_metrics[metric] <= gene_max[metric]]

    count_matrix = count_matrix[:, gene_metrics.index]
    gene_metrics.reset_index(inplace=True)
    gene_metrics.drop('index', axis=1, inplace=True)

    return count_matrix, gene_metrics


# ----------------------------------- #
# filtering genes on relative values  #
# ----------------------------------- #

def filter_rel_genes_on_metric(count_matrix: np.ndarray, 
                                gene_metrics: pd.DataFrame, metric: str, 
                                top: float=None, bottom: float=None) -> tuple:
    """Filters the count_matrix and the gene_metrics based on a metric of the 
    gene metrics. It obtains either the columns associated with the highest 
    values of the defined metric, the lowest values or both.

    Args:
        count_matrix (np.ndarray): matrix of the counts of measured genes in 
            each barcode (barcode x genes).
        gene_metrics (pd.DataFrame): gene metrics (genes, n_barcodes, 
            total_measures, variance).
        metric (str): column of the gene metrics used to subset the data. 
        top (float, optional): if defined, corresponds to the number of columns 
            associated with the highest values of the metric wanted in 
            percentage. Defaults to None.
        bottom (float, optional):  if defined, corresponds to the number of 
            columns associated with the lowest values of the metric wanted in 
            percentage. Defaults to None.

    Returns:
        count_matrix (np.ndarray): filtered matrix of the counts of measured 
            genes in each barcode (barcode x genes).
        gene_metrics (pd.DataFrame): filtered dataframe of gene metrics.
    """

    gene_metrics.sort_values(metric, ascending=top == None, inplace=True)
    gene_metrics = subset_df(gene_metrics, top, bottom)
    
    count_matrix = count_matrix[:, gene_metrics.index]

    gene_metrics.reset_index(inplace=True)
    gene_metrics = gene_metrics.drop('index', axis=1)
    
    return count_matrix, gene_metrics


# ---------------------------- #
#       gene filtering         #
# ---------------------------- #

def filter_genes(gene_names: list, count_matrix: np.ndarray, gene_min: dict={}, 
                    gene_max: dict={}, gene_bottom: dict={}, 
                    gene_top: dict={}) -> tuple:
    """Constructs a dataframe of specified metrics regarding each gene and 
    filters those genes and the count matrix based on the absolute values of 
    those metrics or relative to each other.

    Args:
        gene_names (list[str]): names of the genes counted.
        count_matrix (np.ndarray): counts of the genes (barcode x genes).
        gene_min (dict, optional): metrics (key) for which a minimum value 
            (value) is required for a gene to be included in the downstream 
            analysis. Defaults to {}.
        gene_max (dict, optional): metrics (key) for which a maximum value 
            (value) is required for a gene to be included in the downstream 
            analysis. Defaults to {}.
        gene_top (dict, optional): percentage of genes with the highest value 
            of each metric that are to be used in the downstream analysis.
            Defaults to {}.
        gene_bottom (dict, optional): percentage of genes with the lowest value
            of each metric that are to be used in the downstream analysis.
            Defaults to {}.

    Returns:
        count_matrix (np.ndarray): filtered count matrix with only the genes
            that fall in a given range from the highest values and/or lowest 
            values of each specified metric and with the values within a 
            specified range of values specified by the given metrics.
        gene_metrics (pd.DataFrame): filtered gene metrics with only the genes 
            that fall in a given range from the highest values and/or lowest 
            values of each specified metric and with the values within a 
            specified range of values specified by the given metrics.
    """

    metrics = {'bc_counted': calc_barcode_per_gene, 
                'total_measures': calc_total_measures_per_gene, 
                'variance': calc_var_per_gene, 'mad': calc_mad_per_gene,
                'dispersion': calc_dispersion_per_gene}
    chosen_metrics = (gene_min.keys() | gene_max.keys() | gene_bottom.keys() | 
                        gene_top.keys())
    check_multiple_metrics(metrics, chosen_metrics)

    gene_metrics = calc_gene_metrics(gene_names, count_matrix, chosen_metrics,
                                        metrics)
    count_matrix, gene_metrics = filter_abs_genes(count_matrix, gene_metrics,
                                                    gene_min, gene_max)
    
    return filter_rel_mult_met(count_matrix, gene_metrics, gene_top, 
                                gene_bottom, filter_rel_genes_on_metric)


###############################################################################
# feature selection

# ------------------------------- #
#       correlation methods       #
# ------------------------------- #

def pearson_corr(data: np.ndarray) -> np.ndarray:
    """Calculates the Pearson correlation coefficient between the columns of a 
    given matrix.

    Args:
        data (np.ndarray): matrix of values, where the columns are to be tested
            regarding correlation.

    Returns:
        np.ndarray: matrix of correlation, where the position (i,j) indicates 
            the Pearson correlation coefficient for the columns i and j.
    """

    return np.corrcoef(data, rowvar=False)


def spearman_corr(data: np.ndarray) -> np.ndarray:
    """Calculates the Spearman's Rank correlation coefficient between the 
    columns of a given matrix. Wrapper for the spearmnr function of the scipy 
    package.

    Args:
        data (np.ndarray): matrix of values, where the columns are to be tested
            regarding correlation.

    Returns:
        np.ndarray: matrix of correlation, where the position (i,j) indicates 
            the Spearman's Rank correlation coeficient for the columns i and j.
    """

    return spearmanr(data)[0]


# --------------------------------- #
#  bag removal of correlated genes  #
# --------------------------------- #

def find_corr_cols(bool_corr_matrix: np.ndarray) -> list:
    """Finds the correlated columns of a boolean correlation matrix.

    Args:
        bool_corr_matrix (np.ndarray): boolean correlation matrix where at
            position (i,j) there is a True or a False, indicating that for some
            threshold the columns i and j are correlated or not, respectively.

    Returns:
        corr_centers (list[tuples]): pairs of correlated columns.
    
    Ensures:
        corr_centers: the pairs of correlated columns are sorted on the first
            column and then on the second.
    """

    side = bool_corr_matrix.shape[0]
    corr_centers = []

    for row in np.arange(side-1):
        for col in np.arange(row+1, side):
            if bool_corr_matrix[row][col]:
                corr_centers.append((row, col))
    
    return corr_centers


def getting_bags(corr_centers: list) -> list:
    """Groups the correlated columns on bags, based on whether there is a direct
    or indirect correlation. Example: A and B are correlated; B and C are 
    correlated; A and C are not correlated; A, B, C are all put into the same 
    bag even though A and C are not correlated because there is a thread or 
    correlation network that joins those two.

    Args:
        corr_centers (list[tuples]): pairs of correlated columns.

    Returns:
        bags (list[set]): bags of directly or indirectly correlated columns.
    """

    bags = []
    index = 0
    for x, y in corr_centers:
        if len(bags) == 0:
            bags.append(set((x,y)))
        
        elif x in bags[index]:
            bags[index].add(y)
        
        else:
            bags.append(set((x,y)))
            index += 1

    return bags


def removing_cols_from_bags(bags: list) -> set:
    """Picks one column at random from each bag and removes it from the bag.
    Then joins all the remaining columns of the bags.

    Args:
        bags (list): sets of columns that are correlated either directly or
            indirectly.

    Returns:
        cols_2_rm (set): columns of the bags that weren't picked.
    """

    cols_2_rm = set()
    for bag in bags:
        col = choice(list(bag))
        cols_2_rm |= bag - {col}

    return cols_2_rm


def bag_approach(bool_corr_matrix: np.ndarray) -> list:
    """Takes a bag approach to the removal of correlated/redundant genes. It 
    makes the assumption that if A and B are correlated, B and C are correlated,
    but A and C are not correlated, then the samples of A and B are not 
    correlated, but they represent populations that indeed are. So, A, B and C 
    will be treated as if they were all correlated amongst each other.

    Args:
        bool_corr_matrix (np.ndarray): boolean matrix of correlation where at
            position (i,j) there is either a True or a False indicating if the
            columns i and j are correlated or not, respectively.

    Returns:
        list: indexes of the genes/columns that are to be kept (should not be
            removed).
    """
    corr_centers = find_corr_cols(bool_corr_matrix)
    bags = getting_bags(corr_centers)
    cols_2_rm = removing_cols_from_bags(bags)
    return [i for i in np.arange(bool_corr_matrix.shape[0]) 
                if i not in cols_2_rm]


# ------------------------------- #
#  removal of correlated genes    #
# ------------------------------- #

def select_with_corr(cm: np.ndarray, gene_met: pd.DataFrame, corr_method: str, 
                        threshold: float, keep_method: str) -> tuple:
    """Determines the redundancy of genes based on their correlation scores 
    and removes redundant genes.

    Args:
        cm (np.ndarray): count matrix (barcodes x genes).
        gene_met (pd.DataFrame): gene metrics.
        corr_method (str): method to calculate the correlations of the genes.
        threshold (float): value above which genes are to be considered 
            correlated.
        keep_method (str): method that will decide how correlated genes should
            be eliminated.

    Returns:
        cm (np.ndarray): count matrix without redundant genes.
        gene_met (pd.DataFrame): gene metrics without redundant genes.
    """
    
    avail_corr_methods = {'pearson': pearson_corr, 'spearman': spearman_corr}
    is_chosen_available(avail_corr_methods, corr_method, False)

    avail_keep_methods = {'bag': bag_approach}
    is_chosen_available(avail_keep_methods, keep_method, False)
    
    corr_matrix = avail_corr_methods[corr_method](cm) >= (1 - threshold)
    genes_2_keep = avail_keep_methods[keep_method](corr_matrix)

    gene_met = gene_met.iloc[genes_2_keep]
    gene_met.reset_index(inplace=True)
    gene_met = gene_met.drop('index', axis=1)

    return cm[:,genes_2_keep], gene_met


###############################################################################
# data transformation

# ---------------------------- #
#     coordinate scalling      #
# ---------------------------- #

def min_max_scaling(column: pd.Series):
    """Performs min-max scaling inplace on a column of a dataframe.

    Args:
        column (pd.Series): column of a dataframe to be scaled.
    """

    preprocessing.minmax_scale(column, copy=False)


def scale_coord(barcode_metrics: pd.DataFrame):
    """Performs min-max scaling inplace to the x coordinates and y coordinates
    of the barcodes in the provided dataframe (dataset).

    Args:
        barcode_metrics (pd.DataFrame): dataframe with the coordinates of the
            barcode.
    """

    barcode_metrics.x_coor = min_max_scaling(barcode_metrics.x_coor)
    barcode_metrics.y_coor = min_max_scaling(barcode_metrics.y_coor)


# ---------------------------- #
#  gene count transformation   #
# ---------------------------- #


def robust_scaler(data: np.ndarray):
    """Removes the median and scales the data in accordance with the 
    interquartile range. The scaling is done inplace.

    Args:
        data (np.ndarray): data to be scaled.
    """

    transformer = preprocessing.RobustScaler(copy=False).fit(data)
    transformer.transform(data)


def standardization(data: np.ndarray):
    """Removes the mean and scales the data in such a way that the variance is
    reduced to 1. The standardization is made inplace.

    Args:
        data (np.ndarray): data to be standardized.
    """

    transformer = preprocessing.StandardScaler(copy=False).fit(data)
    transformer.transform(data)


def power_transform(data: np.ndarray, method: str):
    """Performs transfomations on the data using either the Box-Cox or
    the Yeo-Johnson method to make the data have a more normal shape. This 
    transformation is done inplace.

    Args:
        data (np.ndarray): data to be transformed.
        method (str): transformation to be done. Either \'yeo-johnson\' or 
            \'box-cox'.

    Raises:
        ValueError: in case the method chosen is not one of the options.
    """

    if method not in {'yeo-johnson', 'box-cox'}:
        raise ValueError('Methods is not acceptable. Available methods:' +
                            '{\'yeo-johnson\', \'box-cox\'}')
    
    transformer = preprocessing.PowerTransformer(method=method, copy=False)
    transformer.fit(data)
    transformer.transform(data)


def transform_data(data: np.ndarray, transformation: str) -> np.ndarray:
    """Performs scaling or a transformation of the data inplace.

    Args:
        data (np.ndarray): data to be scaled or transformed.
        transformation (str): scaling method or transfomation method to be 
            applied. Options: robust - performs a robust scaling based on the 
            median and interquartile region;  standard - performs 
            standardization of the data (mean = 0, variance = 1); yeo-johnson -
            performs yeo-johnson transformation; box-cox - performs box-cox 
            transformation; log - performs the logaritmization of the data with 
            the natural logarithm; log10 - performs the logaritmization of the
            data with base 10.

    Raises:
        ValueError: in case the transformation chosen is not in the options.

    Returns:
        np.ndarray: transformed/scaled data. 
    """

    transf = {'robust': robust_scaler, 'standard': standardization, 
                'yeo-johnson': partial(power_transform, method=transformation), 
                'box-cox': partial(power_transform, method=transformation),
                'log': np.log, 'log10': np.log10}
    
    is_chosen_available(transf, transformation, False)

    if transformation.startswith("log"):
        return transf[transformation](data)
    
    x, y = data.shape
    data = data.reshape(-1,1)
    transf[transformation](data)
    return data.reshape(x, y)


###############################################################################
# embedded methods