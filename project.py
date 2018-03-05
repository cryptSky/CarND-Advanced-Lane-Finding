import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import Normalize 
import glob
import os
from threshold import color_gradient_threshold, cdmg_threshold
from moviepy.editor import VideoFileClip
from collections import deque
from statistics import mean

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, name, n):
        
        self.name = name
        self.n = n    
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=n)
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = np.array([0, 0, 0])
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        
        self.window_bottom_pos = 250
        self.window_top_pos = 1000
        
        
    #def check_detected()
        
    def process_line(self, recent_x, recent_y, recent_x_fit, current_fit):
        def mean(a):
            return (sum(a) / len(a))
        
        result = self.bestx
        
        if self.detected:
        
            #polynomial coefficients for the most recent fit
            self.allx = recent_x
            self.ally = recent_y
            
            self.recent_xfitted.appendleft(recent_x_fit)
            self.bestx = np.asarray([*map(mean, zip(*self.recent_xfitted))]) #np.mean(self.recent_xfitted, axis=1) # 

            self.diffs = self.current_fit - current_fit
            self.current_fit = current_fit
                        
            result = self.bestx
            
        return result
        
    def params(self):
        print("Line: ", self.name)
        print("detected: ", self.detected)
        print("current fit: ", self.current_fit)
        

class AdvancedLaneRecognizer:
    
    # prepare object points
    nx = 9
    ny = 6
    
    image_points = []
    object_points = []    
    
    left_line = None
    right_line = None
    
    ym_per_pix = 25/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/1000 # meters per pixel in x dimension
    
    def __init__(self, n):
    
        self.left_line = Line("Left", n)
        self.right_line = Line("Right", n)
        
        self.line_base_pos = None
    
        self.objp = np.zeros((self.nx*self.ny, 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)
        
        self.findCorners()        
        
    
    def findCorners(self):
        # Make a list of calibration images
        for fname in glob.glob('camera_cal/calibration*.jpg'):
            img = cv2.imread(fname)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
    
            if ret == True:
                self.image_points.append(corners)
                self.object_points.append(self.objp)
                
                #img = cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
                #plt.imshow(img)
                #plt.show()
                
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.image_points, gray.shape[::-1], None, None)

                
    # Function that takes an image, object points, and image points
    # performs the camera calibration, image distortion correction and 
    # returns the undistorted image
    def undistort(self, img):
        # Use cv2.calibrateCamera() and cv2.undistort()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                
        # undistort
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        
        return undist
        
    def undistort_all(self):
        for fname in glob.glob('test_images/*.jpg'):
            img = cv2.imread(fname)
            undist = self.undistort(img)
            
            path = os.path.join('./test_image_undistorted',fname[fname.rfind('\\')+1:])
            print(path)
            cv2.imwrite(path, undist)
    
    # Define a function that takes an image, number of x and y points, 
    # camera matrix and distortion coefficients
    def perspectiveTransform(self, undist):

        img_size = (undist.shape[1], undist.shape[0])
    
        # Search for corners in the grayscaled image
        corners_before = np.float32([(224, 719),(533,494),(755,494),(1086,719)]) 
        corners_after = np.float32([(230, 719),(224,200),(1086,200),(1080,719)]) 

        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(corners_before, corners_after)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)
    
        # Return the resulting image and matrix
        return warped, M
        
    def perspectiveTransform_all(self):
        for fname in glob.glob('test_image_undistorted/*.jpg'):
            img = cv2.imread(fname)
            transformed, M = self.perspectiveTransform(img)
            
            path = os.path.join('./test_image_transformed',fname[fname.rfind('\\')+1:])
            print(path)
            cv2.imwrite(path, transformed)
            
    def threshold_all(self):
        for fname in glob.glob('test_image_undistorted/*.jpg'):
            img = cv2.imread(fname)
            thresholded = color_gradient_threshold(img)            
            path = os.path.join('./test_image_thresholded',fname[fname.rfind('\\')+1:])        
            cv2.imwrite(path, thresholded*255)
        
        for fname in glob.glob('test_image_transformed/*.jpg'):
            img = cv2.imread(fname)
            thresholded = color_gradient_threshold(img)
            path = os.path.join('./test_image_transformed_thresholded',fname[fname.rfind('\\')+1:])            
            cv2.imwrite(path, thresholded*255)
            
    def find_lane_lines_all(self):
        for fname in glob.glob('test_image_transformed_thresholded/*.jpg'):
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            img = img[:,:]
            print(img.shape)
            self.find_lane_lines(img//255)
            
            #path = os.path.join('./test_image_transformed',fname[fname.rfind('\\')+1:])
            #print(path)
            #cv2.imwrite(path, transformed)
            
    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
        
    #imshape = original_image.shape
    #bottom_left = (int(imshape[1]*0.1),int(imshape[0]*0.95))
    #top_left = (int(imshape[1]*0.4),int(imshape[0]*0.6))
    #top_right = (int(imshape[1]*0.6),int(imshape[0]*0.6))
    #bottom_right = (int(imshape[1]*0.9),int(imshape[0]*0.95))    
    #vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)    
    #region = region_of_interest(edges, vertices)
    
    def find_lane_lines(self, binary_warped, f):

        binary_warped[binary_warped < 1.2] = 0
        binary_warped[binary_warped >= 1.2] = 1

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[:,:], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        from_midpoint = np.argmax(histogram[midpoint:])
        rightx_base = from_midpoint + midpoint        
        diffx = rightx_base - leftx_base
        
        #print(diffx)
        #print(midpoint)
        #print(leftx_base)
        #print(rightx_base)
        #print(np.argmax(histogram[midpoint:]))

        # Choose the number of sliding windows
        nwindows = 20
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 100
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        non_empty_left = []
        non_empty_right = []
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            
            if self.left_line.detected and window == 0:
                leftx_current = int(self.left_line.recent_xfitted[0][binary_warped.shape[0]-1])
            if self.right_line.detected  and window == 0:
                rghtx_current = int(self.right_line.recent_xfitted[0][binary_warped.shape[0]-1])
            
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            midpoint_xleft = (win_xleft_high + win_xleft_low) // 2
            midpoint_xright = (win_xright_low + win_xright_high) // 2
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2) 
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            if len(good_left_inds) > 100:
                left_lane_inds.append(good_left_inds)
                non_empty_left.append(window)
            if len(good_right_inds) > 100:
                right_lane_inds.append(good_right_inds)
                non_empty_right.append(window)
                
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        if len(non_empty_left) == 0 or (np.max(non_empty_left) - np.min(non_empty_left) <= 4):
            self.left_line.detected = False
        else:
            self.left_line.detected = True
                       
        if len(non_empty_right) == 0 or (np.max(non_empty_right) - np.min(non_empty_right) <= 4):
            self.right_line.detected = False
        else:
            self.right_line.detected = True

        
        #if len(empty_left) >= 16 and np.max(empty_left) - np.min(empty_left) <= 5:
        #    for index in range(nwindows):
        #        if index not in empty_right and index in empty_left:
        #            left_lane_inds[index] = right_lane_inds[index] - diffx
        #            
        #if len(empty_right) >= 16 and np.max(empty_right) - np.min(empty_right) <= 5:
        #    for index in range(nwindows):
        #        if index not in empty_left and index in empty_right:
        #            right_lane_inds[index] = left_lane_inds[index] - diffx
        
        #print("non_empty_right: ", non_empty_right)
        #print("non_empty_left: ", non_empty_left)
        #
        #if self.right_line.detected == True and self.left_line.detected == False:
        #    from_right = [elem for elem in non_empty_right if elem not in non_empty_left]
        #    print("from right: ", from_right)
        #    for index in from_right:
        #        new_left_inds = right_lane_inds[non_empty_right.index(index)]
        #        new_left_inds = [index for index in new_left_inds if nonzerox[index] - diffx >= 0]
        #        inds = [i + len(nonzeroy) for i in range(len(new_left_inds))]
        #        nonzeroy = np.append(nonzeroy, nonzeroy[new_left_inds])
        #        nonzerox = np.append(nonzerox, nonzerox[new_left_inds] - diffx)
        #        if len(inds) > 0:
        #            left_lane_inds.append(inds)                
        #
        #    self.left_line.detected = True
        #elif self.left_line.detected == True and self.right_line.detected == False:
        #    from_left = [elem for elem in non_empty_left if elem not in non_empty_right]
        #    print("from left: ", from_left)
        #    for index in from_left:
        #        new_right_inds = left_lane_inds[non_empty_left.index(index)]
        #        new_right_inds = [index for index in new_right_inds if nonzerox[index] + diffx < 1280]
        #        idcs = [i + len(nonzeroy) for i in range(len(new_right_inds))]
        #        nonzeroy = np.append(nonzeroy, nonzeroy[new_right_inds])
        #        nonzerox = np.append(nonzerox, nonzerox[new_right_inds] + diffx)
        #        if len(idcs) > 0:
        #            right_lane_inds.append(idcs)                
        #    #leftx = np.concatenate((leftx, rightx - diffx))
        #    #lefty = np.concatenate((lefty, righty))
        #    self.right_line.detected = True
        
        # Concatenate the arrays of indices
        if len(left_lane_inds) > 0:
            left_lane_inds = np.concatenate(left_lane_inds)
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
        else:
            leftx = [self.left_line.window_bottom_pos]
            lefty = [1279]
            
        if len(right_lane_inds) > 0:
            right_lane_inds = np.concatenate(right_lane_inds)
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
        else:
            rightx = [self.right_line.window_bottom_pos]
            righty = [1279]
       
        ## Fit a second order polynomial to each
        #total_fit = np.polyfit(totaly, totalx, 2)
        #print(total_fit)
        #
        #if len(leftx) >= len(rightx):
        #    total_fit_left = total_fit
        #    total_fit_right = total_fit + [0,0, diffx]
        #else:
        #    total_fit_right = total_fit
        #    total_fit_left = total_fit - [0,0, diffx]
           
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)       
        
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
                
        self.distance_top = right_fitx[0] - left_fitx[0]
        self.distance_bottom = right_fitx[binary_warped.shape[0]-1] - left_fitx[binary_warped.shape[0]-1]
        
        #print ("distance_top : ", self.distance_top)
        #print ("distance_bottom : ", self.distance_bottom)
        
        left_fitx = self.left_line.process_line(leftx, lefty, left_fitx, left_fit)
        right_fitx = self.right_line.process_line(rightx, righty, right_fitx, right_fit)

        #if not check:
        #    self.left_line.detected = False
        #    self.right_line.detected = False
        #    
        #    left_fit = self.left_line.process_line(leftx, lefty, left_fit, center_diff)
        #    right_fit = self.right_line.process_line(rightx, righty, right_fit, center_diff)   
        
        self.calculate_curvature(left_fitx, right_fitx, ploty)
        
        center = midpoint
        lane_center = self.left_line.bestx[binary_warped.shape[0]-1] + (self.right_line.bestx[binary_warped.shape[0]-1] - self.left_line.bestx[binary_warped.shape[0]-1]) // 2        
        center_diff = lane_center - center
        self.line_base_pos = center_diff*self.xm_per_pix
                
        check = self.sanity_check()
        
        #total_fitx = total_fit[0]*ploty**2 + total_fit[1]*ploty + total_fit[2]
        #total_fit_leftx = total_fit_left[0]*ploty**2 + total_fit_left[1]*ploty + total_fit_left[2]
        #total_fit_rightx = total_fit_right[0]*ploty**2 + total_fit_right[1]*ploty + total_fit_right[2]
        #self.fill_with_points(total_fit_leftx, total_fit_rightx)
        
        #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        #out_img = out_img/255
        #axes = f.get_axes()
        #axes[4].set_title('Lane Finding')
        #axes[4].imshow(out_img)
        #axes[4].plot(histogram)
        #axes[4].plot(left_fitx, ploty, color='yellow')
        #axes[4].plot(right_fitx, ploty, color='yellow')
        #axes[5].set_title('Final Image Frame')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)
        #plt.show()
        
        return ploty, left_fitx, right_fitx

            
    def calculate_curvature(self, leftx, rightx, ploty):
    
        y_eval = np.max(ploty)
        # Define conversions in x and y from pixels space to meters

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*self.ym_per_pix, leftx*self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*self.ym_per_pix, rightx*self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*self.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*self.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        self.left_line.radius_of_curvature = left_curverad
        self.right_line.radius_of_curvature = right_curverad
        self.radius_of_curvature = (int)(left_curverad + right_curverad) // 2

        self.diff_curvature_sign = (left_fit_cr[0] < 0 and right_fit_cr[0] > 0) or (left_fit_cr[0] > 0 and right_fit_cr[0] < 0)

        # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 632.1 m    626.2 m
    
    def draw(self, undist, warped, left_fitx, right_fitx, ploty, Minv):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result,"Radius of curvature = {}(m)".format(self.radius_of_curvature),(10,70), font, 2,(255,255,255),2,cv2.LINE_AA)
        if self.line_base_pos < 0:
            cv2.putText(result,"Vehicle is {:1.2f}m left of center".format(np.abs(self.line_base_pos)),(10,150), font, 2,(255,255,255),2,cv2.LINE_AA)
        elif self.line_base_pos > 0:
            cv2.putText(result,"Vehicle is {:1.2f}m right of center".format(np.abs(self.line_base_pos)),(10,150), font, 2,(255,255,255),2,cv2.LINE_AA)
        else:
            cv2.putText(result,'Vehicle is on the center of the lane',(10,150), font, 2,(255,255,255),2,cv2.LINE_AA)
        
        #plt.imshow(result)
        #plt.show()
        return result
    
    def process_frame(self, img):
        undist = self.undistort(img)        
     
        pTransformed, M = self.perspectiveTransform(undist)
        
        #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        #ax1.set_title('Undistorted')
        #ax1.imshow(undist)
        #ax2.set_title('Perspective transformed')
        #ax2.imshow(pTransformed)
        #plt.show()
        
        thresholded, f = cdmg_threshold(pTransformed)

        ploty, left_fitx, right_fitx = self.find_lane_lines(thresholded, f)
        Minv = np.linalg.inv(M)        
        result = self.draw(undist, thresholded, left_fitx, right_fitx, ploty, Minv)        
        return result
        
       
        
    def process_video(self):
        white_output = 'result_harder_challenge_video.mp4'
        ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
        ## To do so add .subclip(start_second,end_second) to the end of the line below
        ## Where start_second and end_second are integer values representing the start and end of the subclip
        ## You may also uncomment the following line for a subclip of the first 5 seconds
        ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
        clip1 = VideoFileClip("harder_challenge_video.mp4")
        white_clip = clip1.fl_image(self.process_frame) #NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)
        white_clip.reader.close()
        white_clip.audio.reader.close_proc()        

    
        
    def sanity_check(self):
        result = True
        
        if np.abs(self.left_line.radius_of_curvature - self.right_line.radius_of_curvature) > 700 or \
            (np.abs(self.left_line.radius_of_curvature - self.right_line.radius_of_curvature) < 700 and np.abs(self.left_line.radius_of_curvature - self.right_line.radius_of_curvature) > 400 and self.diff_curvature_sign):
            result = False
            
        if np.abs(self.distance_top - self.distance_bottom) > 400:
            result = False       
        
        return result
          
    def fill_with_points(self, left_fit, right_fit):
        # Generate some fake data to represent lane-line pixels
        ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
        # For each y position generate random x position within +/-50 pix
        # of the line base position in each case (x=200 for left, and x=900 for right)
        leftx = np.array([left_fit[2] + (y**2)*left_fit[0] + y*left_fit[1] + np.random.randint(-50, high=51) 
                                    for y in ploty])
        rightx = np.array([right_fit[2] + (y**2)*right_fit[0] + y*right_fit[1] + np.random.randint(-50, high=51) 
                                        for y in ploty])
        
        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Plot up the fake data
        mark_size = 3
        plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
        plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(left_fitx, ploty, color='green', linewidth=3)
        plt.plot(right_fitx, ploty, color='green', linewidth=3)
        plt.gca().invert_yaxis() # to visualize as we do the images
        
        self.calculate_curvature(left_fit, right_fit, ploty)
        
    def margin_step(self, binary_warped, margin):
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
        left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
        right_fit[1]*nonzeroy + right_fit[2] + margin)))  
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                    ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                    ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
        # Draw the lane onto the warped blank image
        #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        #result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        #plt.imshow(result)
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)
        #plt.show()
        
    def window_mask(self, width, height, img_ref, center,level):
            output = np.zeros_like(img_ref)
            output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
            return output
        
    def find_window_centroids(self, image, window_width, window_height, margin):
        
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions
        
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
        
        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))
        
        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(image.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,image.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))
    
        return window_centroids
        
    def find_lane_lines_conv(self, warped):
        # window settings
        window_width = 50 
        window_height = 80 # Break image into 9 vertical layers since image height is 720
        margin = 100 # How much to slide left and right for searching        
            
        window_centroids = self.find_window_centroids(warped, window_width, window_height, margin)
        
        # If we found any window centers
        if len(window_centroids) > 0:
        
            # Points used to draw all the left and right windows
            l_points = np.zeros_like(warped)
            r_points = np.zeros_like(warped)
        
            # Go through each level and draw the windows 	
            for level in range(0,len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = self.window_mask(window_width,window_height,warped,window_centroids[level][0],level)
                r_mask = self.window_mask(window_width,window_height,warped,window_centroids[level][1],level)
                # Add graphic points from window mask here to total pixels found 
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
        
            # Draw the results
            template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channel
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
            warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
            output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
        
        # If no window centers found, just display orginal road image
        else:
            output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
        
        # Display the final results
        plt.imshow(output)
        plt.title('window fitting results')
        plt.show() 


if __name__ == "__main__":
    alr = AdvancedLaneRecognizer(10)
    
    #alr.findCorners()
    #alr.undistort_all()
    #alr.perspectiveTransform_all()
    #
    #alr.threshold_all()
    #alr.find_lane_lines_all()
    
    alr.process_video()