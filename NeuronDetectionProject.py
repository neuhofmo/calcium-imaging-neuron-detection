
# coding: utf-8

# # Calcium Imaging - Neuron Detection Project
# Guy Teichman & Moran Neuhof, Winter 2019

from Segmentation.params import *
from Segmentation.utilities import *
from Segmentation.processing_funcs import *
from Segmentation.kmeans import *
from Segmentation.evaluate import *
from Segmentation.cellFluor import *


# ### Loading images and masks


# load the images
files = sorted(glob(imgs_path))
imgs = np.array([imread(f) for f in files])
dims = imgs.shape[1:]

masks = plotregions(regions_file, dims)
summed = np.float64(imgs.sum(axis=0))
# removed artifact
imgf = np.concatenate((imgs[0:2060,:,:], imgs[2250:,:,:]), axis=0)  


# ## Filter each pixel based on the time series

# Creating a list of filters and their parameters
filter_methods = [(ndimage.gaussian_filter1d, 4)]
# Filtering each pixel's change in fluorescence over time
ts_filtered_imgs = filter_timeseries(np.float64(imgf), filter_methods)


# ### Calculate Pearson correlation after filtering time series
neighborCorrcoef = correlate_neighbors(ts_filtered_imgs[0][0])

cont = find_contours(masks.sum(axis=0))  # contours 
show(draw_circles(summed,cont))
corrcoef_norm = norm_data(neighborCorrcoef)  # normalization

# remove background
corrcoef_norm_eq = corrcoef_norm/ndimage.gaussian_filter(corrcoef_norm,sigma=15)
print("Showing correlation image, after equalization - ground truth regions marked")
show(draw_circles(corrcoef_norm_eq, cont))

# removing edges
corrcoef_norm_eq_no_frame = corrcoef_norm_eq.copy()
corrcoef_norm_eq_no_frame[:, 0:18] = 0
corrcoef_norm_eq_no_frame[:, -12:] = 0

# removing background from the original summed image
summed_eq = summed / ndimage.gaussian_filter(summed, sigma=15)

# ### Kmeans and remove background
print("Running Kmeans and removing background")
# Kmeans on the image with no frame
K=16
segmented_data_no_frame, thresholds, centroids = kmeans_with_centroids(corrcoef_norm_eq_no_frame, K)
# Remove background, using kmeans
seg_no_frame = segmented_data_no_frame.copy()
seg_no_frame[seg_no_frame <= thresholds[8]] = 0


# ### Creating boolean image, as seeds for watershed
bool_img = morphology.erosion(seg_no_frame, morphology.disk(1)) > 0


# ### Adding Hit or Miss for donut neurons
print("Running Hit or Miss (donut!)")
hms = norm_data(hitormiss_donut(summed_eq))
show(draw_circles(norm_data(hitormiss_donut(summed_eq)),cont))
hms_labeled = ndimage.label((hms>0.75)&(hms<0.89))[0]
hms_selected = size_selection(hms_labeled, 10, 0)

print("Combining seeds and running watershed...")
# combine seeds of two methods
comb_bool_img = bool_img | morphology.erosion((hms_selected > 0),morphology.disk(1))

# size selection, watershed
size_selected_after_ws = watershed(corrcoef_norm_eq_no_frame , comb_bool_img, 'neuron_detection_final', dims, dial_rad=8, max_neuron_size=21, min_neuron_size=5)
img_w_contours = draw_circles(corrcoef_norm_eq_no_frame, find_contours(size_selected_after_ws))
show(img_w_contours)
print("Neurofinder results:")
print(evaluation('neuron_detection_final'))

# saving image
im = Image.fromarray(img_w_contours)
fname = 'final_regions_190221.jpg'
im.save(fname)
print(f"Image saved: {fname}")

# saving environment
np.save('corrcoef_norm_eq_no_frame_final.npy', corrcoef_norm_eq_no_frame)
np.save('summed_eq_final.npy', summed_eq)
np.save('seg_no_frame_final.npy', seg_no_frame)
print("Intermediate matrix - saved")

# ### Grid search over parameters
# #### DO NOT RUN


# # Grid search on the following:
# # show(hitormiss_donut(new_summed))
# grid_search_res = []
# bool_img_disk_range1 = range(1,2)
# rad_inner_range = range(4,7)
# width_range = range(2,4)
# hms_max_size_range = range(8, 13)
# # thresholds
# hms_labeling_threshold_min_range = range(73, 76)
# hms_labeling_threshold_max_range = range(86, 91)
# bool_img_disk_range2 = range(1,2)

# max_size_range = range(21, 24)
# min_size_range = range(4, 7)
# dial_rad_range = range(6, 10)

# img_tuple = (corrcoef_norm_eq_no_frame, summed_eq)

# total_counts = count_combs([bool_img_disk_range1, rad_inner_range, width_range, hms_max_size_range, 
#                             hms_labeling_threshold_min_range, hms_labeling_threshold_max_range, bool_img_disk_range2,
#                             max_size_range, min_size_range, dial_rad_range,img_tuple])
# print(f"Trying {total_counts} combinations")
# start_time = time.time()
# print(f"Started at {time.strftime('%H:%M:%S', time.gmtime(start_time))}")
# dict_for_bad_eval = {x: 0 for x in ('combined', 'inclusion', 'precision', 'recall', 'exclusion')}  # if eval fails

# counter = 0
# for bool_img_disk_size1 in bool_img_disk_range1:
#     bool_img = morphology.erosion(seg_no_frame, morphology.disk(bool_img_disk_size1)) > 0
#     # hit or miss
#     for rad_inner in rad_inner_range:  # positive radius
#         for width in width_range:  # negative radius
#             if rad_inner + width >= 12:  # bigger than the neuron radius
#                 continue  # ignore
#             hms = norm_data(hitormiss_donut(summed_eq, rad_inner, width))
#             # show(draw_circles(hms, cont))
#             # hms_with_cont = draw_circles(hms, cont)
#             for min_threshold in hms_labeling_threshold_min_range:
#                 min_threshold /= 100
#                 for max_threshold in hms_labeling_threshold_max_range:
#                     max_threshold /= 100
#                     hms_labeled = ndimage.label(((min_threshold < hms) & (hms < max_threshold)))[0]  # between thresholds
#                     # show(hms_labeled)
#                     for hms_max_size in hms_max_size_range:
#                         hms_selected = size_selection(hms_labeled, hms_max_size, 0)  # selecting smaller neurons
#                         hms_selected_bool = hms_selected > 0
#                         for bool_img_disk_size2 in bool_img_disk_range2:
#                             bool_img2 = morphology.erosion(hms_selected_bool, morphology.disk(bool_img_disk_size2))
#                             comb_bool_img = bool_img | bool_img2
#                             for dial_rad in dial_rad_range:
#                                 for max_neuron_size in max_size_range:
#                                     for min_neuron_size in min_size_range:
#                                         for i, img in enumerate(img_tuple):
#                                             json_fname = "grid_search_json"
#                                             size_selected_after_ws = watershed(img, comb_bool_img, json_fname, dims, dial_rad=dial_rad, max_neuron_size=max_neuron_size, min_neuron_size=min_neuron_size)
#                                             if size_selected_after_ws is None:  # if size selection didn't keep anything
#                                                 d = {"bool_img_disk_size1": bool_img_disk_size1,
#                                                  "rad_inner": rad_inner,
#                                                  "width": width,
#                                                  "min_threshold": min_threshold,
#                                                  "max_threshold": max_threshold,
#                                                  "hms_max_size": hms_max_size,
#                                                  "bool_img_disk_size2": bool_img_disk_size2,
#                                                  "dial_rad": dial_rad,
#                                                  "max_neuron_size": max_neuron_size,
#                                                  "min_neuron_size": min_neuron_size,
#                                                  "img": i,
#                                                  "evaluation": dict_for_bad_eval
#                                                 }
#                                             else:
#                                                 # img_w_contours = draw_circles(corrcoef_norm_eq_no_frame, find_contours(size_selected_after_ws))
#                                                 # show(img_w_contours)
#                                                 evaluation_res = evaluation(json_fname)
                                                
#                                                 d = {"bool_img_disk_size1": bool_img_disk_size1,
#                                                      "rad_inner": rad_inner,
#                                                      "width": width,
#                                                      "min_threshold": min_threshold,
#                                                      "max_threshold": max_threshold,
#                                                      "hms_max_size": hms_max_size,
#                                                      "bool_img_disk_size2": bool_img_disk_size2,
#                                                      "dial_rad": dial_rad,
#                                                      "max_neuron_size": max_neuron_size,
#                                                      "min_neuron_size": min_neuron_size,
#                                                      "img": i,
#                                                      "evaluation": evaluation_res
#                                                     }
#                                             grid_search_res.append(d)
#                                             os.remove(fix_json_fname(json_fname))
#                                             # counter
#                                             counter += 1
#                                             if counter % 100 == 0:
#                                                 elapsed_time = time.time() - start_time                                                
#                                                 print(f"Progress: {counter}/{total_counts} ({counter/total_counts * 100:.2f}%). Elapsed: {time.strftime('%H:%M:%S', elapsed_time)}")
#                                             # once in 100:
#                                             if counter % 1000 == 0:
#                                                 # with open('res_last.pickle', 'wb') as f:  # used to be
#                                                 with open('res_last_2.pickle', 'wb') as f:
#                                                     # Pickle the 'data' dictionary using the highest protocol available.
#                                                     pickle.dump(grid_search_res, f, pickle.HIGHEST_PROTOCOL)
#                                                     print("Results pickled")



# print(f"Finished at {time.strftime('%H:%M:%S', time.gmtime(start_time))}")
# with open('res_last_2.pickle', 'wb') as f:
#     # Pickle the 'data' dictionary using the highest protocol available.
#     pickle.dump(grid_search_res, f, pickle.HIGHEST_PROTOCOL)
#     print("Results pickled")

print("Done with neuron detection!")
print("***************************")

# ## Plotting fluorescence

# ### Plotting the results achieved by our pipeline on the same dataset
# 

print("Plotting cell fluorescence for each cell over time")

# plot fluorescence
activity = plotCellFluorescence(ts_filtered_imgs[0][0], 'neuron_detection_final', figsize=(12,30), neuron_lim=None)

print("Plotting cell fluorescence for some of the cells over time")
# plot fluorescence
activity = plotCellFluorescence(ts_filtered_imgs[0][0], 'neuron_detection_final', neuron_lim=(0,60))

