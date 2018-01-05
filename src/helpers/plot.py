import math
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Grid helper funcions:

def getGridFor(n_items, n_cols = 4, hspace = 0.2, wspace = 0.05):
    return getGrid(math.ceil(n_items / n_cols), n_cols, hspace, wspace)

def getGrid(n_rows, n_cols = 4, hspace = 0.2, wspace = 0.05):    
    plt.figure(random.randrange(0, 1000), figsize=(n_cols * 8, n_rows * 6))

    return gridspec.GridSpec(n_rows, n_cols, hspace, wspace)