from Segmentation.utilities import *
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.misc import imread
from scipy import ndimage, signal
from skimage import morphology, feature, exposure
import neurofinder
import cv2 as cv
import ast
from PIL import Image


def tomask(coords, dims):
    mask = np.zeros(dims,dtype=bool)
    coords=np.array(coords)
    mask[coords[:,0],coords[:,1]] = 1
    return mask

# load the regions (training data only)
def plotregions(filename, dims):
    with open(filename+'.json') as f:
        regions = json.load(f)
    
    masks = np.array([tomask(s['coordinates'], dims) for s in regions])  
    # generate a list of all masks based on the regions
    return masks


#turning the maskes into the json file
# def mask_to_json(mask, filename, prelabeled=False):
#     """Receive a mask and a file name and save to a json file"""
#     if not prelabeled:
#         labeled, n = ndimage.label(mask)  # creates one image with numerical labels for each neuron
#     else:  # prelabeled 
#         labeled = mask  # labeled mask, from watershed
#         n = np.max(labeled)  # highest label
    
#     myRegions = []  # initialize
#     # We don't need size selection anymore, so the loop looks like this now:
#     elem_gen = (np.nonzero(labeled == i) for i in range(1, n+1))
#     selected_elem_gen = (elem for elem in elem_gen if len(elem[0]) != 0 and len(elem[1]) != 0)
#     # elem_gen = (elem for elem in (np.nonzero(labeled == i) for i in range(1, n+1)) if len(elem[0]) != 0 and len(elem[1]) != 0)
    
#     myRegions = [{"id":i, "coordinates":[[x[0], x[1]] for x in zip(elem)]} for i, elem in enumerate(selected_elem_gen)]
#     print(myRegions)
#     # saving as json
#     json.dump(myRegions, open(fix_json_fname(filename),'w'))

    # for elem in elem_gen:
    #     myRegions.append({"id":elem, "coordinates":list(zip(elem))}) 
        # for i,j in zip(xelem, yelem):
        #     myRegions[-1]["coordinates"].append([int(i),int(j)])

    # # for elem in range(1, n+1):
    # #     xelem, yelem = np.nonzero(labeled == elem)
        
    # #     if len(xelem) == 0 or len(yelem) == 0:     # skip empty elements
    # #         continue
        
    #     myRegions.append({"id":elem, "coordinates":[]})    # add coordinates as json
    #     for i,j in zip(xelem, yelem):
    #         myRegions[-1]["coordinates"].append([int(i),int(j)])
    
    
#turning the maskes into the json file
def mask_to_json(mask, filename, prelabeled=False):
    """Receive a mask and a file name and save to a json file"""
    if not prelabeled:
        labeled, n = ndimage.label(mask)  # creates one image with numerical labels for each neuron
    else:  # prelabeled 
        labeled = mask  # labeled mask, from watershed
        n = np.max(labeled)  # highest label
    
    myRegions = []  # initialize
    # We don't need size selection anymore, so the loop looks like this now:

    for elem in range(1, n+1):
        xelem, yelem = np.nonzero(labeled == elem)
        
        if len(xelem) == 0 or len(yelem) == 0:     # skip empty elements
            continue
        
        myRegions.append({"id":elem, "coordinates":[]})    # add coordinates as json
        for i,j in zip(xelem, yelem):
            myRegions[-1]["coordinates"].append([int(i),int(j)])
    
    # saving as json
    json.dump(myRegions, open(fix_json_fname(filename),'w'))


def plotRandomNeuron(imgs, masks):
    """Chooses and plots a random neuron and a random image out of the imgs matrix"""
    plt.figure(figsize=(12,12))
    i = np.random.randint(0, len(masks))
    plt.subplot(1, 2, 1)
    plt.imshow(imgs[i], cmap='gray')
    plt.title(f'All Neurons - sum of all images ({len(imgs)} timepoints)')
    plt.subplot(1, 2, 2)
    plt.imshow(masks[i], cmap='gray')
    plt.title(f'Neuron #{i} - mask')
    plt.tight_layout()


def bandPassFilter(img, radIn=50, radOut=10000, plot=False):
    """Receive image, inner radius size, outer radius size.
    Return an image filtered with a disk, using FFT."""
    
    # FFT
    fft=np.fft.fftshift(np.fft.fft2(img))
    # shape
    x,y=np.shape(fft)
    xg, yg = np.ogrid[-x//2:x//2, -y//2:y//2]
    
    # define  filter disk
    inner_circle_pixels = xg**2 + yg**2 <= radIn^2
    outer_circle_pixels=xg**2 + yg**2 <= radOut^2
    filter_disk = np.ones_like(inner_circle_pixels)
    filter_disk[np.invert(outer_circle_pixels)] = 0
    filter_disk[inner_circle_pixels] = 0
    # filter
    fftfilt = fft * filter_disk

    ifft = np.fft.ifft2(fftfilt)
    img_filt = np.abs(ifft)

    if plot:  # plot the filtered image
        plt.figure()
        plt.imshow(img_filt, cmap='gray') 
        plt.figure()
        plt.imshow(np.log10(np.power(np.abs(fftfilt),2)))  # plot the disk
    
    return img_filt

### 
def sobel(summed):  
    """Perform sobel on an image, and return it"""
    sx = ndimage.sobel(summed, axis=0, mode='constant')
    sy = ndimage.sobel(summed, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    return sob

# 
def filter_timeseries(imgs, filter_methods):
    """Receive a 3d matrix, filter over time using a filter method."""

    # examples:
    # ndimage.gaussian_filter1d, 1
    # ndimage.gaussian_filter1d, 4
    # gaussian_laplace, 3
    # maximum_filter1d
    
    # initialize a zeros matrix for each of the filtering methods
    filtered_imgs = [[np.zeros(imgs.shape), filter_methods[i][0], filter_methods[i][1]] for i in range(len(filter_methods))] 
    print(f"Running {len(filter_methods)} timelapse filtering methods.")
    
    for method_num, (filter_method, param) in enumerate(filter_methods):  # run each of the methods
        filtered_imgs[method_num][0] = filter_method(imgs, param, axis=0)
    
    return filtered_imgs  # [img, method, param]


# Not used in the final version
def hitormiss(img, radP=5, radN=2):
    """Hit or miss filter"""
    r1 = radP  # radius of hit-or-miss-circle (positive values)
    r2 = radP+radN  # radius of negative values in hit-or-miss-circle (to emphasize edges)
    disk = np.int8(morphology.disk(r1))
    largedisk = np.int8(np.zeros([r2*2+1,r2*2+1]))
    largedisk[(r2-r1):r1*2+(1+r2-r1),(r2-r1):r1*2+(1+r2-r1)] = disk
    negdisk = np.int8(morphology.disk(r2))
    negdisk *= -1
    negdisk += largedisk*2
    largedisk[largedisk==0] = -1
    
    conv = signal.convolve2d(exposure.adjust_gamma(sobel(img),gamma=0.8), negdisk, mode='full')
    (cx,cy) = np.shape(conv)
    conv = conv[(cx-512)//2:cx-(cx-512)//2,(cy-512)//2:cx-(cy-512)//2]
    plt.imshow(conv,cmap='gray')
    plt.title('hitormiss')
    return conv


def find_contours(labeled_image):
    """Receive a labeled image and return the contours of the image, marked by -1's"""
    # save image shape
    x,y = labeled_image.shape
    img = labeled_image.copy()  # save a copy
    
    # iterate over image
    for i in range(x):
        for j in range(y):
            neighbor_pairs = [[i, j+1], [i,j-1], [i+1, j], [i-1, j]]  # save neighbors
            if labeled_image[i,j] == 0:  # if the value of the labeled image is 0
                neighbors = []  # initialize
                for pair in neighbor_pairs:  # save the pairs
                    try:
                        neighbors.append(labeled_image[pair[0], pair[1]])  # add the value of the neighbor
                    except IndexError:  # if it's a corner, or the frame of the image
                        continue  # skipping frame
                if np.max(neighbors) > 0:  # if one of the neighbors has a labeled value
                    img[i,j] = -1  # mark this contour position with -1 
    return img


def draw_circles(image, markers):
    """Receive an image, and a markers array, and returns a RGB image with the contours labeled in red."""
    # turning image to img,  RGB
    RGB_img = np.zeros([512,512,3], dtype='uint8')
    # keep the same image in all 3 colors
    for i in range(3):
        RGB_img[:,:,i] = np.ndarray.astype(norm_data(image)*255,'uint8')
    # save a copy  of the image in red
    red_image = np.ndarray.astype(norm_data(image)*255,'uint8') 
    red_image[markers == -1] = 255 # Contours are red
    RGB_img[:,:,0] = red_image  # combine image with contours with image without
    return RGB_img


def size_selection(labeled, max_neuron_size=21, min_neuron_size=4, coef=2, verbose=False):
    """ 
    labeled: image with label of each neuron
    max_neuron_size: the maximal neuron size accepted
    min_neuron_size: the minimal neuron size accepted
    coef: the x/y or y/x maximal ratio
    verbose: True writes down the number of kept/removed neurons.
    """
    new_labeled = labeled.copy()
    n = np.max(new_labeled)  # number of neurons
    counter = 0
    for elem in range(1, n+1):  # iterate over neurons
        xelem, yelem = np.nonzero(new_labeled == elem)  # keep the positions of the element
        try:
            neuron_max_x_y = (np.max(xelem) - np.min(xelem), np.max(yelem) - np.min(yelem))    # ( size x, size y)
            # if neuron is too big or too small
            if np.max(neuron_max_x_y) > max_neuron_size or np.min(neuron_max_x_y) < min_neuron_size or is_not_round(neuron_max_x_y, coef):  # or the neuron shape is not round
                # TODO: add shape selection here (is_round)
                new_labeled[new_labeled == elem] = 0
                counter += 1
        except ValueError:
            continue  
    
    kept = n - counter
    if verbose:
        print(f"removed {counter} neurons. Kept {kept} neurons.")

    new_labeled[new_labeled == -1] = 0
    
    if kept == 0 and verbose:
        print("No neurons kept! ignoring it.")
        return None
    else:
        return new_labeled


def is_not_round(neuron_max_x_y, coef=2):
    """Receives the length/width of the object, and returns True if the object is not round.
    coef: the x/y or y/x maximal ratio."""
    x_len, y_len = neuron_max_x_y
    if (x_len/y_len) > coef or (y_len/x_len) > coef:
        return True
    else:
        return False


def watershed(image,mask, filename, dims, dial_rad=9, max_neuron_size=21, min_neuron_size=4, coef=2):
    """
    Receives an image, a boolean mask containing the seeds for the watershed algorithm, the json filename to save,
    the dimensions of the image and an optional dialation radius and max/min neuron sizes for size selection,
    as well as the length/width ratio (coef).
    """ 
    # sure foreground
    sure_fg = np.uint8(mask)
    sure_bg = np.uint8(morphology.binary_dilation(mask,morphology.disk(dial_rad)))
    
    # initialize
    img = np.zeros([512,512,3], dtype='uint8')
    img[:,:,0] = np.ndarray.astype(norm_data(image)*255,'uint8')
    
    # unknown area
    unknown = cv.subtract(sure_bg, sure_fg)
    
    # check connected components and label
    ret, markers = cv.connectedComponents(sure_fg)
    del ret
    # make backgorund 1, and unknown 0
    markers += 1
    markers[unknown==1] = 0
    # run watershed
    markers_after_WS = cv.watershed(img,markers)
    
    size_selected_markers = size_selection(markers_after_WS, max_neuron_size, min_neuron_size, coef)
    if size_selected_markers is not None:  # if size selection kept anything
        mask_to_json(size_selected_markers, filename, True)
        return size_selected_markers
    else:
        return None


def calculate_change_in_time(imgf, gaussian_sigma, disk_size, threshold):
    
    # calculate change in time
    diff = np.abs(np.diff(imgf, axis=0))  # derivative
    # diff = np.abs(np.diff(imgf_gaussian_4, axis=0))  # derivative
    diffmax = diff.max(axis=0) / ndimage.gaussian_filter(diff.max(axis=0),sigma=gaussian_sigma)  #
    # normalizing the difference (between each pixel, over time points)
    # divided by the max value of blurry image
    # The blurry image is the background
    diffmaxopen = morphology.opening(diffmax,morphology.disk(disk_size))  # after opening
    diffmax_bool = norm_data(diffmaxopen) > threshold  # normalize and take the
    return diffmax, diffmax_bool


def save_image(img, filename):
    """Saves black and white image (img) as file (filename)"""
    im = Image.fromarray(draw_circles(ndimage.gaussian_filter(img,sigma=15), np.zeros((512,512))))
    
    # add suffix if needed
    if filename.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'bmp']:
        filename = filename+'.jpg'
    
    im.save(filename)
    print(f"{filename} saved")



def hitormiss_donut(img, rad_inner=5, width=2):
    """
    Hit or miss- donut shaped, for less active neurons.
    rad_inner: radius of negative values
    width: radius of positive values (around the edges)
    """
    r1 = rad_inner  #radius of hit-or-miss-circle (negative values)
    r2 = rad_inner + width  # radius of positive values in hit-or-miss-circle (to emphasize edges)
    
    disk = np.int8(morphology.disk(r1))
    frame_with_disk = np.int8(np.zeros([r2*2+ 1, r2*2+1]))  #  frame
    frame_with_disk[(r2-r1):r1*2 + (1+r2-r1), (r2-r1):r1*2 + (1+r2-r1)] = disk  # disk in the center of frame
    
    # outer circle
    larger_disk = np.int8(morphology.disk(r2))  # larger disk, in the shape of frame
    donut = larger_disk -2 * frame_with_disk
    
    # convolve the disk with the image
    conv = signal.convolve2d((norm_data(img) - 0.5), donut, mode='full') 
    (cx,cy) = np.shape(conv)
    conv = conv[(cx-512)//2:cx-(cx-512)//2, (cy-512)//2:cx-(cy-512)//2]  # resize
    return conv


mem_d_corrcoef = {}  # initialize
def corrcoef_val_per_couple(i, j, r, c, data, d=mem_d_corrcoef):
    """Calculate the correlation value per each two positions.
    Saves it to dict d for memoization"""
    if (i,j,r,c) in d:
        return d[(i,j,r,c)]
    else:
        val = np.corrcoef(data[:,i,j], data[:,r,c])[0,1]  
        # keep one of the corr matrix
        d[(i,j,r,c)] = val  # 
        return val


def correlate_neighbors(data):
    """Calculate correlation matrix between adjacent neurons over time."""
    ## 8 neighbors
    counter = 0  # intialize for progress
    neighborCorrcoef = np.zeros([512,512],dtype='float64')  # the matrix we write into
    n = 512*512  # total pixels
    for i in range(512):    # going over all the rows
        for j in range(512):    # going over all the columns
            cnt = 0  #how many neighbors did we successfully test (for regular pixel is 8, for corner is 3, for edge is 5)
            newval = 0  # the sum of all of the correlations, that we will use at the value of the specific pixel we test
            # build neighbor list
            neighbor_pairs = [(i+a,j+b) for a,b in ((-1, 0), (1, 0), (0, -1), (0, 1), (1,1), (-1,1), (-1,-1), (1,-1))]
            
            # check neighbors
            for pair in neighbor_pairs:  #going over each neighbor's coordinate (row-column) pair
                r,c = pair # unpacking the row-column values from thte pair
                if r>=0 and c>=0: #make sure no negative indices
                    try:
                        newval += corrcoef_val_per_couple(i, j, r, c, data)
                        # add to newval the sum of the multiplication of the two times series of the original pixel and it's neighbor, divided by the multiplication of the average values of them both
                        cnt += 1  #if we reached this we successfully tested the neighbor
                    except IndexError:  #this is for corners and edges
                        continue
            # normalize value (for edges)
            neighborCorrcoef[i,j] = newval / cnt  # divide the total value by the number of neighbors that were tested and summed, and then write into the final matrix
            counter += 1  # increment for progress
            if counter % 10000 == 0:  # progress print
                print(f"Progress: {counter}/{n} ({counter/(n) * 100:.2f}%)")
    return neighborCorrcoef

