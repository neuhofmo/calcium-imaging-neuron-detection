import numpy as np
import matplotlib.pyplot as plt

### This library performs Kmeans and separates different values of brightness

def create_threshold(data, K):
    """Initialize thresholds based on value range"""
    min_val = np.min(data)
    max_val = np.max(data)
    bins = np.linspace(min_val, max_val, num=K, endpoint=False)  # value histogram with equal spacing
    bins=bins[1:]  # setting up bin threshold K-1
    return bins


def compute_segment_values(data, thresholds):
    """Return a list of all values of the image pixels, divided into segments (based on their thresholds)"""
    segments = []
    # first segment:
    segment = data[(0 <= data) & (data < thresholds[0])]
    segments.append(segment)
    
    # all thresholds
    for i in range(1, len(thresholds)):
        segment = data[(thresholds[i-1] <= data) & (data < thresholds[i])]
        segments.append(segment)
    
    # last_segment
    segment = data[(thresholds[-1] <= data) & (data <= 1)]
    segments.append(segment)
    
    return segments


def compute_centroids(segments, thresholds):
    """Compute the centroids of K segments"""
    centroids = []  # initialize list
    thresholds = [0] + [thresh for thresh in thresholds] + [1]
    for i,segment in enumerate(segments):
        if len(segment) == 0:  # segment is empty
            centroid = (thresholds[i+1] - thresholds[i]) / 2  # mean between two thresholds
            centroids.append(centroid)
        else:
            centroids.append(np.mean(segment))
    return centroids


def compute_new_thresholds(centroids):
    # we now have K centroids. 
    # The thresholds should be midway between each two centroids
    thresholds = []
    for i in range(1, len(centroids)):
        threshold = centroids[i-1] + ((centroids[i] - centroids[i-1])/ 2)
        thresholds.append(threshold)
    
    return thresholds


def centroids_stable(new_centroids, old_centroids, diff_thresh = 1e-5):
    """Returns True of the centroids are stable."""
    diff_centroid_arr = np.abs(np.array(new_centroids) - np.array(old_centroids))
    for cent_diff in diff_centroid_arr:
        if cent_diff > diff_thresh:
            # too much of a difference
            return False
    return True  # stayed stagnant


def segment_image(data, thresholds, centroids):
    """Segment the image for presentation"""
    segmented_data = data.copy()  # new version
    # first segment:
    segmented_data[(0 <= data) & (data < thresholds[0])] = centroids[0]
    # all thresholds
    for i in range(1, len(thresholds)):
        segmented_data[(thresholds[i-1] <= data) & (data < thresholds[i])] = centroids[i]
    # last_segment
    segmented_data[(thresholds[-1] <= data) & (data <= 1)] = centroids[-1]
    return segmented_data


def kmeans(data, K):  # deprecated
    "Implement K-means with K clusters"
    # initialize
    thresholds = create_threshold(data, K)  # K-1 thresholds
    segments = compute_segment_values(data, thresholds)  # should be K segments
    centroids = compute_centroids(segments, thresholds)  # should be K centroids
    
    # new centroids
    old_centroids = [0] * K
    
    # iterate
    while not centroids_stable(centroids, old_centroids):  # as long as the centroids are not stable
        # reset thresholds to be midway between cluster centers
        thresholds = compute_new_thresholds(centroids)
        segments = compute_segment_values(data, thresholds)  # should be K segments
        old_centroids = centroids
        centroids = compute_centroids(segments, thresholds)  # should be K centroids
        
    
    # color image based on centroid values:
    
    segmented_data = segment_image(data, thresholds, centroids)
    return segmented_data


def kmeans_with_centroids(data, K):
    "Implement K-means with K clusters, return centroids"
    # initialize
    thresholds = create_threshold(data, K)  # K-1 thresholds
    segments = compute_segment_values(data, thresholds)  # should be K segments
    centroids = compute_centroids(segments, thresholds)  # should be K centroids
    
    # new centroids
    old_centroids = [0] * K
    
    # iterate
    while not centroids_stable(centroids, old_centroids):  # as long as the centroids are not stable
        # reset thresholds to be midway between cluster centers
        thresholds = compute_new_thresholds(centroids)
        segments = compute_segment_values(data, thresholds)  # should be K segments
        old_centroids = centroids
        centroids = compute_centroids(segments, thresholds)  # should be K centroids
        
    
    # color image based on centroid values:
    
    segmented_data = segment_image(data, thresholds, centroids)
    return segmented_data, thresholds, centroids

def run_and_plot_kmeans(data, last_K=16):
    """Run K means algorithm and plot it, with various Ks"""
    plt.subplots(2,3, figsize=(16,16))

    plt.subplot(231)
    plt.imshow(data, cmap='gray')
    plt.title(f"Before segmentation")

    plt.subplot(232)
    K=2
    segmented_data, thresholds, centroids = kmeans_with_centroids(data, K)
    plt.imshow(segmented_data, cmap='gray')
    plt.title(f"After segmentation, Kmeans with K={K}")

    plt.subplot(233)
    K=3
    segmented_data, thresholds, centroids = kmeans_with_centroids(data, K)
    plt.imshow(segmented_data, cmap='gray')
    plt.title(f"After segmentation, Kmeans with K={K}")

    plt.subplot(234)
    K=6
    segmented_data, thresholds, centroids = kmeans_with_centroids(data, K)
    plt.imshow(segmented_data, cmap='gray')
    plt.title(f"After segmentation, Kmeans with K={K}")

    plt.subplot(235)
    K=10
    segmented_data, thresholds, centroids = kmeans_with_centroids(data, K)
    plt.imshow(segmented_data, cmap='gray')
    plt.title(f"After segmentation, Kmeans with K={K}")

    plt.subplot(236)
    K = last_K
    segmented_data, thresholds, centroids = kmeans_with_centroids(data, K)
    plt.imshow(segmented_data, cmap='gray')
    plt.title(f"After segmentation, Kmeans with K={K}")

    plt.tight_layout()
    plt.show()

    return segmented_data, thresholds, K
