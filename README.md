
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/distorted_undistorted.png "Distorted-Undistorted"
[image2]: ./output_images/undistorted_ptransformed.png "Road Transformed"
[image3]: ./output_images/pipeline_sample.png "Pipeline Sample"
[video1]: ./result_project_video.mp4 "Project Video"
[video2]: ./result_challenge_video.mp4 "Challange Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
#### I have two files in my solution: [threshold.py](threshold.py) and [project.py](project.py). First one is used for thresholding input image based on different parameters like color, gradient magnitude, gradient direction.
#### In [project.py](project.py) file you can find  ```AdvancedLaneRecognizer ``` class, which does all the job of finding lane lines and creating final videos. Also there is ```Line ``` class for tracking lane line data.

### Camera Calibration

The code for this step is located at the beginning of ```AdvancedLaneRecognizer ``` class.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `self.objp` is just a replicated array of coordinates, and `self.object_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `self.image_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
This is done inside `def findCorners(self)` method of  ```AdvancedLaneRecognizer ``` class.

I then used the output `self.object_points` and `self.image_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
This is done inside `def undistort(self, img)` method of  ```AdvancedLaneRecognizer ``` class.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image, here I use functions from [threshold.py](threshold.py) file. Main function where everything happens is  `cdmg_threshold`.

 After some time of figuring out what combination of color and gradient thresholds will work good most of the time, I decided to use several approaches. As the first component I used combination of color gradient from HLS color space and x gradient direction. I've figured out that as lane lines are in most cases yellow and white, their upper and lower bounds in the HLS color space are the following: white - from[  0, 185,   0] to [255, 255, 255] and yellow - from [ 0,   0, 40] to [255, 255, 255]. This mask is then merged with thresholded x direction gradient. You can find this functionality inside  `color_gradient_threshold ` function. Then I compute gradient magnitude threshold and gradient directions threshold in `mag_thresh` and `dir_thresh` functions. Finally, in `cdmg_threshold` I combine these 3 masks into one by using weighted sum of those, you can see it on line 151 of [threshold.py](threshold.py). Because color is the most important here it has weight of 2, magnitude and direction have weight of 1. Later, when I search for the lines, inside `find_lane_lines` method from ```AdvancedLaneRecognizer ``` class. In [project.py](project.py) file, I mask resulting weights by thresholding them by 1.2: everything greater than 1.2 becomes one, evrything lower than 1.2 becomes zero (lines 233, 234 from [project.py](project.py) file).
 
Here's an pipeline example of my output for this step. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspectiveTransform`, which appears in lines 149 through 164 in the file [project.py](project.py).  The `perspectiveTransform` function takes undistorted image as input.   I chose the hardcode the source and destination points in the following manner:

```python
corners_before = np.float32([(224, 719),(533,494),(755,494),(1086,719)]) 
corners_after = np.float32([(230, 719),(224,200),(1086,200),(1080,719)]) 
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 224, 719      | 230, 719        | 
| 533,494      | 224,200    |
| 755,494     | 1086,200     |
| 1086,719      | 1080,719       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image2]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To identify lane line pixels, firstly I check histogram of binary mask I've got from previous step. Approximate position of left lane line will correspond to historgam peak to the left from image center, similarly approximate position of the right lane line will be somewhere near histogram peak on the right side of the image.
Then, from the bottom to the top of the image I seek point which are close to histogram peak using small windows. I use 100 pixels as window size and 20 windows. Then, if I find more than 100 pixels in current window, I count them as good candidates and move center of the image towards mean of those points and move to the next window. This process repeats untill all 20 window are processed. 
On the next step I take all those good candidates from previous step and find the polynom of 2nd order to fit them using `np.polyfit` function. It gives me coefficients of that 2nd order polynom - it is actually the equation of curve which corresponds to lane line.

If there are not enough windows to determine good line, for example in my case if the distance between lowest and highest window is less than 4, I assume that for this line we cannot find good curve equation, then I use previous best fit. After each step, to smooth lane lines curves, I use average from the last N successful attempts(10 in my case). If on the previous frame we have succesfuly found the line, the starting point for windows in the next frame is not the histogram peak but on the lowest point of previously found lane line curve. 

To see example of windows you can check out pipeline image above.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I used approximate scale from pixels to meters to find scaled to meters version of the lane curve and used formula to calculate radius of the curve(here is the tutorial https://www.intmath.com/applications-differentiation/8-radius-curvature.php)
Here are those scaling parameters:
```python
    ym_per_pix = 25/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/1000 # meters per pixel in x dimension
```
You can chech the implementation in the function ```calculate_curvature``` (lines  452-473 from [project.py](project.py))

To find position of the vehicle with respect to the center, I just found the difference between center of the image and the middle point between found lane lines (lines  425-428 from [project.py](project.py))

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

To see example of windows you can check out pipeline image above. Open it in new window or tab to see it better.

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]

Here is a [link to challenge video result][video2]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced some problems when trying to adapt my first solution to challenge videos. I was able successfuly build the mask which work good on both project and challenge video. To work on harder challenge video I do not have much time left, so I decided to leave it as it is, but you can check out how my pipeline works on that too [here](./result_harder_challenge_video.mp4). For example, to track curves with higher angles better we will probably need different algorithm, for example, which will start processing window not on histogram peak but on some high value and then pass the highest value and end on some similar hight value on the other side of its peak.
Different conditions require more sophisticated analisys and techniques. I can imgine that some machine learning techniques could help to predict lane lines position on difficult conditions. This could be great advantage to use some highly accurate predictions for this task.