from preprocessing import *

###############################################################################

if __name__ == '__main__':
    import time
    import resource

    time_start = time.perf_counter()

    ################################
    # actual code
    ################################
    
    a, b, c = load_data("../../data/Puck_190926_06_combined.csv", 
                min_genes_bc=1, max_var_bc=6)
    
    print(b.head(n=5))
    print()
    print(c.head(n=5))

    a, b = filter_top_barcodes(a, b, 'variance', 5, 5)
    print()
    print(b)

    a, c = filter_genes(a, c, 5, 1000, min_var=0.02, max_var=8)
    print()
    print(c.head(n=5))


    ################################
    time_elapsed = (time.perf_counter() - time_start)
    memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    print ("%5.1f secs %5.1f MByte" % (time_elapsed,memMb))