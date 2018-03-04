import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    
    return binary_output

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude 
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(np.absolute(sobelx)**2 + np.absolute(sobely)**2)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1
    
    return binary_output

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.sqrt(np.absolute(sobelx)**2)
    abs_sobely = np.sqrt(np.absolute(sobely)**2)
    
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    
    scaled_sobel = np.uint8(255*grad_dir/np.max(grad_dir))

    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir > thresh[0]) & (grad_dir < thresh[1])] = 1
    
    return binary_output

def select_white_yellow(image, color_scheme):
    converted = cv2.cvtColor(image, color_scheme)
    # white color mask
    lower = np.uint8([  0, 185,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 0,   0, 40])
    upper = np.uint8([ 255, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)
    
def color_gradient_threshold(img, color_threshold=(100, 255), gradient_threshold=(100, 255)):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    
    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    
    s_channel = select_white_yellow(img, cv2.COLOR_BGR2HLS)
    s_channel = cv2.cvtColor(s_channel, cv2.COLOR_HLS2RGB)
    s_channel = cv2.cvtColor(s_channel, cv2.COLOR_RGB2GRAY)
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= color_threshold[0]) & (s_channel <= color_threshold[1])] = 1
    
    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= gradient_threshold[0]) & (scaled_sobel <= gradient_threshold[1])] = 1
    
    # Threshold color channel
    #s_binary = np.zeros_like(s_channel)
    #s_binary[(s_channel >= color_threshold[0]) & (s_channel <= color_threshold[1])] = 1
    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    ## Plotting thresholded images
    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    #ax1.set_title('Stacked thresholds')
    #ax1.imshow(color_binary)
    #ax2.set_title('Combined S channel and gradient thresholds')
    #ax2.imshow(combined_binary, cmap='gray')
    #plt.show()
    
    return combined_binary    

    
def cdmg_threshold(image, color_threshold=(100, 255), direction_thresh=(0.0, 0.9), magnitude_thresh=(2, 255), gradient_threshold=(10, 255)):
    
    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    
    # Apply each of the thresholding functions
    color_grad_binary = color_gradient_threshold(image, color_threshold, gradient_threshold)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=magnitude_thresh)
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=direction_thresh)
    
    combined = np.zeros_like(color_grad_binary)
    #combined[(((dir_binary == 1) | (mag_binary == 1)) | (color_grad_binary == 1))] = 1
    
    combined = (2*color_grad_binary + mag_binary + dir_binary) / 3
    #combined = color_grad_binary
    #stacked_combined = np.dstack(( color_grad_binary, mag_binary, dir_binary))
    
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20,10), squeeze=False)
    #ax1.set_title('Stacked color-gradient thresholds')
    #ax1.imshow(color_grad_binary, cmap='gray')
    #
    #ax2.set_title('Gradient magnitude thresholds')
    #ax2.imshow(mag_binary, cmap='gray')
    #
    #ax3.set_title('Gradient direction thresholds')
    #ax3.imshow(dir_binary, cmap='gray')
    #
    #ax4.set_title('Combined')
    #ax4.imshow(combined*255, cmap='gray')
    
    return combined, f