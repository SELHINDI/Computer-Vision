"""
Sammy Elhindi 9/15/24
CS 4391 Homework 2 Programming: Part 1&2 - mean and gaussian filters
Implement the linear_local_filtering() and gauss_kernel_generator() functions in this python script
"""

import cv2
import numpy as np
import math
import sys
 
def linear_local_filtering(
    img: np.uint8,
    filter_weights: np.ndarray,
) -> np.uint8:
    """
    Homework 2 Part 1
    Compute the filtered image given an input image and a kernel 
    """

    img = img / 255
    img = img.astype("float32") # input image
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image
    kernel_size = filter_weights.shape[0] # filter kernel size
    sizeX, sizeY = img.shape

    # filtering for each pixel
    for i in range(kernel_size // 2, sizeX - kernel_size // 2):
        for j in range(kernel_size // 2, sizeY - kernel_size // 2):

            # Todo: For current position [i, j], you need to compute the filtered pixel value: img_filtered[i, j] 
            # using the kernel weights: filter_weights and the neighboring pixels of img[i, j] in the kernel_sizexkernel_size local window
            # The filtering formula can be found in slide 3 of lecture 6

            # ********************************
            filtered_value = 0.0

            for k in range(-(kernel_size // 2), kernel_size // 2 + 1):
                for l in range(-(kernel_size // 2), kernel_size // 2 + 1):
                    # get  the neighboro pixel  from the img
                    neighbor_pixel = img[i + k, j + l]
                    # Get  corresponding filter weight from  kernel
                    weight = filter_weights[k + kernel_size // 2, l + kernel_size // 2]

                    
                    # aply the filter (multiply and accumulate)
                    filtered_value += weight * neighbor_pixel



            #update filter
            img_filtered[i, j] = filtered_value

            # ********************************

        # 

    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered

 
def gauss_kernel_generator(kernel_size: int, spatial_variance: float) -> np.ndarray:
    """
    Homework 2 Part 2
    Create a kernel_sizexkernel_size gaussian kernel of given the variance. 
    """
    # Todo: given variance: spatial_variance and kernel size, you need to create a kernel_sizexkernel_size gaussian kernel
    # Please check out the formula in slide 15 of lecture 6 to learn how to compute the gaussian kernel weight: g[k, l] at each position [k, l].
    kernel_weights = np.zeros((kernel_size, kernel_size))
    #min hone
    center = kernel_size // 2  # The center index of the kernel
    sigma_s = spatial_variance  # Spatial variance

    # Constant part of the Gaussian equation
    constant = 1 / (2 * np.pi * sigma_s**2)

    #make Gaussian kernel
    for k in range(kernel_size):

        for l in range(kernel_size):
            # (center - kernel)^2

            distance_squared = (k - center) ** 2 + (l - center) ** 2
            # = Gaussian function weights
            kernel_weights[k, l] = constant * np.exp(-distance_squared / (2 * sigma_s**2))

    # normalize 
    kernel_weights /= np.sum(kernel_weights)


    return kernel_weights
 
if __name__ == "__main__":
    img = cv2.imread("data/img/butterfly.jpeg", 0) # read gray image
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA) # reduce image size for saving your computation time
    cv2.imwrite('results/im_original.png', img) # save image 
    
    # Generate Gaussian noise
    noise = np.random.normal(0,0.6,img.size)
    noise = noise.reshape(img.shape[0],img.shape[1]).astype('uint8')
   
    # Add the generated Gaussian noise to the image
    img_noise = cv2.add(img, noise)
    cv2.imwrite('results/im_noisy.png', img_noise)
    
    # mean filtering
    box_filter = np.ones((7, 7))/49
    img_avg = linear_local_filtering(img_noise, box_filter) # apply the filter to process the image: img_noise
    cv2.imwrite('results/im_box.png', img_avg)

    # Gaussian filtering
    kernel_size = 7  
    spatial_var = 15 # sigma_s^2 
    gaussian_filter = gauss_kernel_generator(kernel_size, spatial_var)
    gaussian_filter_normlized = gaussian_filter / (np.sum(gaussian_filter)+1e-16) # normalization term
    im_g = linear_local_filtering(img_noise, gaussian_filter_normlized) # apply the filter to process the image: img_noise
    cv2.imwrite('results/im_gaussian.png', im_g)