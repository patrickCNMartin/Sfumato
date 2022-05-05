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

#def convertToNumpy(counts,coordinates):
#    #Extracting dimensions
#    barcodes = counts.shape[1]
#    genes = counts.shape[0]
#    # Estimate number and rows and extraction locations
#    nrow = coordinates["ycoord"].max() - coordinates["ycoord"].min()
#    nrow = nrow.astype(np.int64)
#    ncol = coordinates["xcoord"].max() - coordinates["xcoord"].min()
#    ncol = ncol.astype(np.int64)
#    coordinates[["xcoord","ycoord"]] = coordinates[["xcoord","ycoord"]].apply(lambda coord : coord - coord.min())
#    coordinates[["xcoord","ycoord"]] = coordinates[["xcoord","ycoord"]].astype(np.int64)
#    #Build place holder template
#    arr = np.zeros((nrow+1,ncol+1,genes))
#    for l in range(coordinates.shape[0]):
#        bar = coordinates.loc[l,"barcodes"]
#        nonZeroCounts = counts[bar] != 0
#        arr[coordinates.loc[l,"ycoord"],coordinates.loc[l,"xcoord"],nonZeroCounts] = counts.loc[counts[bar] != 0,bar]
#    ####
#    return arr


#-----------------------/Converting/-------------------------------------------#

#counts = sorted(glob.glob('./*.txt.gz'))
#beads = sorted(glob.glob('./*csv'))

#for files in range(len(counts)):
    #print(files)
    #cloc = loadCounts(counts[files])
    #print("Counts loaded")
    #bloc = pd.read_csv(beads[files])
    #filename = beads[files][2:16] + '.npy'
    #arr = convertToNumpy(cloc,bloc)
