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