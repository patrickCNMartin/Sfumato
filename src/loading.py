################################################################################
##########################    Gradient Analysis     ############################
################################################################################
#----------------------/Loading modules/---------------------------------------#
import numpy as np
import pandas as pd
import os
import gzip
import glob
import re
from scipy.sparse import csr_matrix


#-----------------------/Functions/--------------------------------------------#

def loadCounts(counts):
    cloc = []
    with gzip.open(counts,'rb') as f:
        for line in f:
            cloc.append(line)
    ##################
    for line in range(len(cloc)):
        cloc[line] = cloc[line].decode("utf-8")
        cloc[line] = re.sub("\n","",cloc[line])
        cloc[line] = re.sub("\r","",cloc[line])
        cloc[line] = cloc[line].split("\t")
    ##################
    cloc = pd.DataFrame(cloc)
    searchfor = ['GENE', 'Row']
    headerloc = cloc[0]
    headerloc = headerloc.str.contains('|'.join(searchfor))
    #headerloc = cloc.where(cloc.iloc[0,:]=="GENE")
    header = cloc.loc[headerloc.values,:]
    header = header.iloc[0]
    cloc = cloc.loc[(~headerloc.values),:]
    cloc.columns = header
    return cloc
