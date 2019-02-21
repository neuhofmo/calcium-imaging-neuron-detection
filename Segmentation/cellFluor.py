from Segmentation.params import regions_file
from Segmentation.utilities import *
from Segmentation.processing_funcs import tomask
import json

def plotCellFluorescence(imgs, filename, sampling_frequency=7, figsize=(12,12), neuron_lim=None): 
    """Receive the images and coordinated filename, and plots the fluorescence over time. 
    Optional arguments:
    sampling_frequency: In our case, it's 7 Hz
    neuon_lim: a tuple of the range of neuorns (out of the whole dataset) to plot. If None, plot all."""
    with open(filename+'.json') as f:   # read the json file again
        regions = json.load(f)
    
    dims = imgs.shape[1:]
    cells = np.array([tomask(s['coordinates'], dims) for s in regions])  # regions
    activity = np.zeros([np.shape(cells)[0], np.shape(imgs)[0]])  # initialize timeseries
    # activity (cell x timepoints) - measure activity for each cell (cell) for each timepoint (img, n)
    
    for cell, cellCoord in enumerate(cells):  # for each cell
        p = len(cellCoord) # number of pixels in ROI
        Ft = np.sum(imgs[:,cellCoord],axis=1) / p # value of all pixels divided by number of pixels
        #Fxsmoothed=0 # calculate Fx smoothed
        F0 = np.percentile(Ft,10) #calculate F0  # normalized to 10th percentile
        Rt = (Ft-F0) / F0
        activity[cell,:] = Rt
        # sum of all flourescence values in the cell coordinates - for each of the timepoints
    
    plt.figure(figsize=figsize)
    for cell in range(np.shape(cells)[0]):
        plt.plot(np.linspace(0, np.shape(imgs)[0] / sampling_frequency, np.shape(imgs)[0]), activity[cell,:] + cell)  
    plt.title("Fluorescence of recognized cells")
    plt.xlabel("$t$ [s]")
    plt.ylabel("Cell ID")
    if neuron_lim:
        min_neuron, max_neuron = neuron_lim
        plt.ylim(min_neuron, max_neuron)  # limit to only a certain amount of neurons
    sns.despine()
    plt.show()
    return activity