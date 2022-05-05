################################################################################
##########################    Gradient Analysis     ############################
################################################################################
#--------------------------------/ imports /-----------------------------------#
import numpy as np
import random

#------------------------------/ Simulations /---------------------------------#

class simTissue:
    def __init__(self, n, structType):
        self.grid = np.zeros((n,n))
        self.gridSize = n
        self.structType = structType
    def seedGrid(self,n_seeds, n_cells):
        canvas = self.grid
        x = random.sample(range(self.grid), k = n_seeds).astype(np.int64)
        y = random.sample(range(self.grid), k = n_seeds).astype(np.int64)
        canvas[x,y] = random.choices(range(n_cells),k=n_seeds).astype(np.int64)
        return canvas
