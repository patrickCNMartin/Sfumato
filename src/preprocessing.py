import csv

import numpy as np
import pandas as pd

from math import ceil
from typing import Generator
from functools import reduce, partial
from sklearn import preprocessing
from scipy.stats import spearmanr

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
#       barcode metrics        #
# ---------------------------- #

def row_n_genes(tags: dict, gene_counts: np.ndarray):
    """Calculates the number of different genes that were counted at least once,
    in a given barcode.

    Args:
        tags (dict): tags of the row (barcode, x coordinate, y coordinate).
        gene_counts (np.ndarray): counts of each gene for the current barcode.
    """

    tags['counted_genes'] = np.sum(gene_counts > 0)


def row_gene_counts(tags: dict, gene_counts: np.ndarray):
    """Calculates the total number of genes that were counted in a given 
    barcode.

    Args:
        tags (dict): tags of the row (barcode, x coordinate, y coordinate).
        gene_counts (np.ndarray): counts of each gene for the current barcode.
    """

    tags['total_counts'] = np.sum(gene_counts)


def row_gene_var(tags: dict, gene_counts: np.ndarray):
    """Calculates the variance of the gene count in a given barcode.

    Args:
        tags (dict): tags of the row (barcode, x coordinate, y coordinate).
        gene_counts (np.ndarray): counts of the genes for a given barcode.
    """

    tags['variance'] = np.var(gene_counts)


def calc_row_metrics(tags: dict, gene_counts: np.ndarray) -> dict:
    """Calculates the row wise metrics.

    Args:
        tags (dict): tags of the row (barcode, x coordinate, y coordinate).
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


# ------------------------------- #
# filtering with barcode metrics  #
# ------------------------------- #

def filter_barcodes(tags: dict, min_genes: int=None, max_genes: int=None, 
                    min_measures: int=None, max_measures: int=None, 
                    min_var: float=None, max_var: float=None) -> bool:
    """Determines if a given row (a given barcode) follows the specified 
    conditions regarding the gene measurements.

    Args:
        tags (dict): metrics regarding a single barcode.
        min_genes (int, optional): Minimum number of genes required to be 
            detected by the barcode, for it to be included. Defaults to None.
        max_genes (int, optional): Maximum number of genes possible to be 
            detected by the barcode, before it is excluded. Defaults to None.
        min_measures (int, optional): Minimum amount of total measures required
            to be detected by the barcode, for it to be included. Defaults to 
            None.
        max_measures (int, optional): Maximum possible number of total measures 
            of genes detected by the barcode, before it is excluded. Defaults to 
            None.
        min_var (float, optional): Minimum amount of variance in the measures of 
            the genes in a single barcode, required for it to be included. 
            Defaults to None.
        max_var (float, optional): Maximum amount of variance possible in the 
            measures of the genes in a single barcode, before it is excluded. 
            Defaults to None.

    Returns:
        bool: flag indicating whether the measures of a barcode should be 
            included in the dataframe for analysis.
    """
    
    if min_genes and tags['counted_genes'] < min_genes:
        return False
    if max_genes and tags['counted_genes'] > max_genes:
        return False
    
    if min_measures and tags['total_counts'] < min_measures:
        return False
    if max_measures and tags['total_counts'] > max_measures:
        return False

    if min_var and tags['variance'] < min_var:
        return False
    if max_var and tags['variance'] > max_var:
        return False
    
    return True


# ---------------------------- #
#        gene metrics          #
# ---------------------------- #

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


def calc_gene_metrics(gene_names: list, 
                        gene_counts: np.ndarray) -> pd.DataFrame:
    """Constructs a dataframe with the metrics regarding each gene. Namely, it
    includes the number of barcodes the measure each gene, the total number of
    measures of each gene and the variance in the measures of each gene.

    Args:
        gene_names (list): names of the genes in the count matrix.
        gene_counts (np.ndarray): count matrix of barcodes x genes.

    Returns:
        genes (pd.DataFrame): metrics regarding each gene. 
    """
    
    genes = pd.DataFrame({'genes': gene_names})
    genes['n_barcodes'] = calc_barcode_per_gene(gene_counts)
    genes['total_measures'] = calc_total_measures_per_gene(gene_counts)
    genes['variance'] = calc_var_per_gene(gene_counts)

    return genes


# ---------------------------- #
#   data and metrics loader    #
# ---------------------------- #

def load_data(filename: str, delimiter: str=",", skip_fst_col: bool=True,
                min_genes_bc: int=None, max_genes_bc: int=None, 
                    min_measures_bc: int=None, 
                    max_measures_bc: int=None, min_var_bc: float=None, 
                    max_var_bc: float=None) -> tuple:
    """Loads the data and calculates some basic metrics regarding the data. It
    also uses the metrics to do some preliminary filtering of the data.

    Args:
        filename (str): name and path of the file that contains the data.
        delimiter (str, optional): delimiter of the fields of the datafile. 
            Defaults to ",".
        skip_fst_col (bool, optional): flag indicate whether to consider the 
            first column or not. Defaults to True.
        min_genes_bc (int, optional): Minimum number of genes required to be 
            detected by the barcode, for it to be included. Defaults to None.
        max_genes_bc (int, optional): Maximum number of genes possible to be 
            detected by the barcode, before it is excluded. Defaults to None.
        min_measures_bc (int, optional): Minimum amount of total measures 
            required to be detected by the barcode, for it to be included. 
            Defaults to None.
        max_measures_bc (int, optional): Maximum possible number of total 
            measures of genes detected by the barcode, before it is excluded. 
            Defaults to None.
        min_var_bc (float, optional): Minimum amount of variance in the measures 
            of the genes in a single barcode, required for it to be included. 
            Defaults to None.
        max_var_bc (float, optional): Maximum amount of variance possible in the 
            measures of the genes in a single barcode, before it is excluded. 
            Defaults to None.

    Returns:
        count_matrix (np.ndarray): counts of the measures of the 
            barcodes x genes.
        barcode_met (pd.DataFrame): row-wise metrics (metrics regarding each 
            barcode).
        gene_met (np.ndarray): column-wise metrics (metrics regarding each 
            gene).
    """
    
    data = readr_generator(filename, delimiter)
    gene_names, data = obtain_gene_names(data, skip_fst_col)
    
    def joiner(acc: tuple, row: tuple) -> tuple:
        tags, gene_counts = sep_tag_counts(row, skip_fst_col)
        bc_metrics = calc_row_metrics(tags, gene_counts)
        clean = filter_barcodes(bc_metrics, min_genes_bc, max_genes_bc, 
                                min_measures_bc, max_measures_bc, min_var_bc, 
                                max_var_bc)
        return (acc[0] + ([bc_metrics] if clean else []),
                acc[1] + ([gene_counts] if clean else []))

    barcode_met, count_matrix = reduce(joiner, data, ([], []))
    count_matrix = np.array(count_matrix)
    gene_met = calc_gene_metrics(gene_names, count_matrix)

    return count_matrix, pd.DataFrame(barcode_met), gene_met


###############################################################################
# filtering

# ---------------------------- #
#       helper functions       #
# ---------------------------- #

def check_usage(possible_metrics: set, metric: str, top: float=None, 
                bottom: float=None):
    """Checks if the filtering functions are being correctly used.

    Args:
        possible_metrics (set): names of the columns that can be used for 
            filtering.
        metric (str): selected column.
        top (float, optional): if there is a value, it indicates that the top n 
            values (where n is the value of top) of the chosen column are 
            wanted. Defaults to None.
        bottom (float, optional): if there is a value, it indicates that the 
            bottom n values (where n is the value of bottom) of the chosen 
            column are wanted. Defaults to None.

    Raises:
        ValueError: this error is raised if the column chosen to perform the 
            filtering on is not supposed to be filtered.
        ValueError: this error is raised if there is no top value and no bottom
            value.
    """
    
    if metric not in possible_metrics:
        raise ValueError(f"You chose the following metric: '{metric}'." +
                            "The metric chosen must correspond to one of the" +
                            f" following metrics: {possible_metrics}")
    
    if not top and not bottom:
        raise ValueError("You have to select to either take the top x values" +
                    ", for the bottom x values. You have chosen neither.")


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


# ---------------------------- #
#      rowwise filtering       #
# ---------------------------- #

def filter_rel_barcodes(count_matrix: np.ndarray, 
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
    
    Raises:
        ValueError: if an unavailable metric was used or neither the top or 
            bottom values have been defined.
    """
    
    check_usage({'counted_genes', "total_counts", "variance"}, metric, top, 
                bottom)

    barcode_metrics.sort_values(metric, ascending=top == None, inplace=True)
    barcode_metrics = subset_df(barcode_metrics, top, bottom)
    
    count_matrix = count_matrix[barcode_metrics.index, :]
    
    barcode_metrics.reset_index(inplace = True)
    barcode_metrics = barcode_metrics.drop('index', axis=1) 

    return count_matrix, barcode_metrics


# ---------------------------- #
#     columnwise filtering     #
# ---------------------------- #

def filter_rel_genes(count_matrix: np.ndarray, gene_metrics: pd.DataFrame, 
                    metric: str, top: float=None, bottom: float=None) -> tuple:
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
    
    Raises:
        ValueError: if an unavailable metric was used or neither the top or 
            bottom values have been defined.
    """
    
    check_usage({'n_barcodes', 'total_measures', 'variance'}, metric, top, 
                bottom)
    gene_metrics.sort_values(metric, ascending=top == None, inplace=True)
    gene_metrics = subset_df(gene_metrics, top, bottom)
    
    count_matrix = count_matrix[:, gene_metrics.index]

    gene_metrics.reset_index(inplace=True)
    gene_metrics = gene_metrics.drop('index', axis=1)
    
    return count_matrix, gene_metrics


def filter_genes(count_matrix: np.ndarray, gene_metrics: pd.DataFrame, 
                    min_bc: int=None, max_bc: int=None, min_measures: int=None, 
                    max_measures: int=None, min_var: float=None, 
                    max_var: float=None) -> tuple:
    """Filters the genes based on the pure value of the metrics (as posed to 
    based on the comparison of values).

    Args:
        count_matrix (np.ndarray): matrix of the counts of measured genes in 
            each barcode (barcode x genes).
        gene_metrics (pd.DataFrame): gene metrics (genes, n_barcodes, 
            total_measures, variance).
        min_bc (int, optional): minimum number of barecodes for a gene to be 
            considered. Defaults to None.
        max_bc (int, optional): maximum number of barecodes that can detect that 
            gene for it is to be removed. Defaults to None.
        min_measures (int, optional): minimum number of total measurements of 
            the gene for it to be included. Defaults to None.
        max_measures (int, optional): maximum number possible of total 
            measurements before the gene is to be excluded. Defaults to None.
        min_var (float, optional): minimum variance required for a gene to be 
            included. Defaults to None.
        max_var (float, optional): maximum variance possible of gene before it 
            is to be excluded. Defaults to None.

    Returns:
        count_matrix (np.ndarray): filtered matrix of the counts of measured 
            genes in each barcode (barcode x genes).
        gene_metrics (pd.DataFrame): filtered dataframe of gene metrics.
    """
    
    if min_bc:
        gene_metrics = gene_metrics[gene_metrics.n_barcodes >= min_bc]
    if max_bc:
        gene_metrics = gene_metrics[gene_metrics.n_barcodes <= max_bc]
    
    if min_measures:
        gene_metrics = gene_metrics[gene_metrics.total_measures >= min_measures]
    if min_measures:
        gene_metrics = gene_metrics[gene_metrics.total_measures <= max_measures]
    
    if min_var:
        gene_metrics = gene_metrics[gene_metrics.variance >= min_var]
    if max_var:
        gene_metrics = gene_metrics[gene_metrics.variance <= max_var]

    count_matrix = count_matrix[:, gene_metrics.index]
    gene_metrics.reset_index(inplace=True)
    gene_metrics.drop('index', axis=1, inplace=True)

    return count_matrix, gene_metrics


###############################################################################
# feature selection

def pearson_corr(data: np.ndarray) -> np.ndarray:
    return np.corrcoef(data, rowvar=False)


def spearman_corr(data: np.ndarray) -> np.ndarray:
    return spearmanr(data)[0]


def select_with_corr(data: np.ndarray, corr_method: str, threshold: float):
    methods = {'pearson': pearson_corr, 'spearman': spearman_corr}
    if corr_method not in methods:
        raise ValueError(f"The corr_method chosen \'{corr_method}\' is not " +
                            "of the available methods. Available methods are" +
                            f" {list(methods.keys())}.")
    
    corr_matrix = methods[corr_method](data) >= (1 - threshold)


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
    
    if transformation not in transf:
        raise ValueError('Transformation is not acceptable. Available ' +
                            f'transformations: {list(transf.keys())}')

    if transformation.startswith("log"):
        return transf[transformation](data)
    
    x, y = data.shape
    data = data.reshape(-1,1)
    transf[transformation](data)
    return data.reshape(x, y)


###############################################################################
# embedded methods