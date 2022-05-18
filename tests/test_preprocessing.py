from ..src.preprocessing import *

###############################################################################

if __name__ == '__main__':
    import time
    import resource

    time_start = time.perf_counter()

    ################################
    # actual code
    ################################
    
    a, b, c = load_data("../../data/Puck_190926_06_combined.csv", 
                min_genes_bc=5, min_var_bc=2)

    a, c = filter_genes(a, c, min_bc=5)
    a, c = filter_rel_genes(a, c, metric='variance', top=30)
    print(pearson_corr[:5,:10])


    ################################
    time_elapsed = (time.perf_counter() - time_start)
    memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    print ("%5.1f secs %5.1f MByte" % (time_elapsed,memMb))