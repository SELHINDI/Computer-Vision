"""
Sammy Elhindi 9/15/24

CS 4391 Homework 2 Programming: Part 3 - bilateral filter
Implement the bilateral_filtering() function in this python script
"""
 
import cv2
import numpy as np
import math

def bilateral_filtering(
    img: np.uint8,
    spatial_variance: float,
    intensity_variance: float,
    kernel_size: int,
) -> np.uint8:
    """
    Homework 2 Part 3
    Compute the bilaterally filtered image given an input image, kernel size, spatial variance, and intensity range variance
    """

    img = img / 255
    img = img.astype("float32")
    img_filtered = np.zeros(img.shape) # Placeholder of the filtered image
    
    # Todo: For each pixel position [i, j], you need to compute the filtered output: img_filtered[i, j]
    # step 1: compute kernel_sizexkernel_size spatial and intensity range weights of the bilateral filter in terms of spatial_variance and intensity_variance. 
    # step 2: compute the filtered pixel img_filtered[i, j] using the obtained kernel weights and the neighboring pixels of img[i, j] in the kernel_sizexkernel_size local window
    # The bilateral filtering formula can be found in slide 15 of lecture 6
    # Tip: use zero-padding to address the black border issue.

    # ********************************
   # 1/2  kernel size
    half_k = kernel_size // 2

    # pad
    padded_img = np.pad(img, ((half_k, half_k), (half_k, half_k)), mode='constant')


    # find  Gaussian weights
    spatial_gaussian = np.zeros((kernel_size, kernel_size), dtype="float32")
    for i in range(kernel_size):

        for j in range(kernel_size):
            distance_squared = (i - half_k) ** 2 + (j - half_k) ** 2
            spatial_gaussian[i, j] = np.exp(-distance_squared / (2 * spatial_variance))
   
    #  bilateral filter
    
    
    for i in range(half_k, img.shape[0] + half_k):
        for j in range(half_k, img.shape[1] + half_k):
            
            # extract teh local region (kernel)
            
            local_region = padded_img[i - half_k:i + half_k + 1, j - half_k:j + half_k + 1]
            # get intensity Gaussian weights
            intensity_gaussian = np.exp(-((local_region - padded_img[i, j]) ** 2) / (2 * intensity_variance))
            # combine  spatial &  intensity weights
            
            combined_weights = spatial_gaussian * intensity_gaussian
            # ormalize
            normalization_factor = np.sum(combined_weights)

            # use  filter , new val
            img_filtered[i - half_k, j - half_k] = np.sum(combined_weights * local_region) / normalization_factor

    # Convert back to the original intensity range

    # ********************************
    
    
    img_filtered = img_filtered * 255
    img_filtered = np.uint8(img_filtered)
    return img_filtered

 
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
    
    # Bilateral filtering
    spatial_variance = 30 # signma_s^2
    intensity_variance = 0.5 # sigma_r^2
    kernel_size = 7
    img_bi = bilateral_filtering(img_noise, spatial_variance, intensity_variance, kernel_size)
    cv2.imwrite('results/im_bilateral.png', img_bi)