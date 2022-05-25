import sys
sys.path.append('../src')

from preprocessing import *

###############################################################################

if __name__ == '__main__':
    import time
    import resource

    time_start = time.perf_counter()

    ################################
    # actual code
    ################################
    
    g = loader("../../data/Puck_190926_06_combined.csv")[1]
    cm, bc_met = filter_barcodes(g, {'counted_genes': 5, 'variance': 0.1})
    print(cm.shape)

    # a, c = filter_genes(a, c, min_bc=3, min_var=10)
    # # print(a.shape)
    
    # a, c = filter_rel_genes(a, c, metric='variance', top=60)
    # # print(a.shape)
    # # print(spearman_corr(a)[:5,:10])


    ################################
    time_elapsed = (time.perf_counter() - time_start)
    memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    print ("%5.1f secs %5.1f MByte" % (time_elapsed,memMb))