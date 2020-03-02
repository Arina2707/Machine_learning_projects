# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:41:33 2020

@author: Arina27
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from skimage.io import imread
from skimage import img_as_float
import pylab

image=img_as_float(imread("picture.jpg"))
#pylab.imshow(image)

#to matrix
w, h, d = image.shape
pixels=pd.DataFrame(np.reshape(image, (w*h, d)), columns=["R", "G", "B"])

#learning
def cluster_pixels(pixels: pd.DataFrame, n_clusters):
    pixels=pixels.copy()
    model=KMeans(n_clusters, init="k-means++", random_state=241)
    pixels["cluster"]=model.fit_predict(pixels)
    return pixels

def mean_median_image(pixels: pd.DataFrame):
    means=pixels.groupby("cluster").mean().values
    mean_pixels=np.array([means[c] for c in pixels["cluster"]])
    mean_image=np.reshape(mean_pixels, (w, h, d))
    
    medians=pixels.groupby("cluster").median().values
    median_pixels=np.array([medians[d] for d in pixels["cluster"]])
    median_image=np.reshape(median_pixels, (w, h, d))
    
    return mean_image, median_image

def psnr(im1, im2):
    mse=np.mean((im1-im2)**2)
    return 10.0*np.log10(1.0/mse)

for n in range(1, 21):
    print(f"Clustering: {n}")
    
    cpixels = cluster_pixels(pixels, n)
    mean_image, median_image = mean_median_image(cpixels)

    
    psnr_mean, psnr_median = psnr(image, mean_image), psnr(image, median_image)
    print(f"PSNR (mean):", psnr_mean, "PSNR (median):", psnr_median) 
    if psnr_mean > 20 or psnr_median > 20:
        print(str(n))
        break
