CarND · T1 · P4 · Lane Lines Detection Project Writeup
======================================================


[//]: # (Image References)

[image0]: ./output/images/001%20-%20Example%20Output.png "Example Output"
[image1]: ./output/images/002%20-%20Undistorted%20Image.png "Undistorted Image"
[image2]: ./output/images/003%20-%20Undistorted%20Road%20Image.png "Undistorted Road Image"
[image3]: ./output/images/004%20-%20Binary%20Road.png "Binary Road"
[image4]: ./output/images/005%20-%20Warped%20Road.png "Warped Road"
[image5]: ./output/images/006%20-%20Fitting%20a%20Polynomial%20From%20Histogram%20Peaks.png "Fitting a Polynomial From Histogram Peaks"
[image6]: ./output/images/007%20-%20Fitting%20a%20Polynomial%20From%20Previous%20One.png "Fitting a Polynomial From Previous One"


Project Goals
-------------

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


Rubric Points
-------------

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.  


### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in `src/notebooks/Camera Calibration.ipynb`, which uses the methods `findChessboardCorners` and `calibrate_camera` that can be found in `src/helpers/cameraCalibration.py`.

`findChessboardCorners` will read all the camera calibration images from the directory we indicate using its params. For each of the calibration images it will:

- Read the image and resize it if indicated (added this when I tested it with my own camera's images, which were way bigger than the one provided for the project).

- Convert it from BGR to grayscale.

- Try to find the indicated number of columns and rows on it, in this case 9 and 6, respectively, using `cv2.findChessboardCorners`.

- Prepare `obj_points` (object points), which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.

  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
  
- It will plot and/or save the images with the chessboard corners drawn of it if indicated using the correcponding params.

Next, `obj_points` and `img_points` are passed to `calibrate_camera`, which uses them to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function and store the results (`mtx`, `dist`, `rvecs` and `tvecs`, although just the former 2 are used) in a pickle file, so that I don't need to calculate them again.

Lastly, in that same Notebook, `src/notebooks/Camera Calibration.ipynb`, I used `mtx` and `dist` to apply a distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistorted Image][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![Undistorted Road Image][image2]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

All these transforms to convert a raw image into a bird's eye view perspective are implemented in `src/helpers/imagePreprocessingPipeline.py`, in a function `pipeline`, that takes an already undistorted image `img` and a transform matrix `M` as arguments.

The steps that follow are:

1. Generate grayscale and HLS versions of that initial image to use them as inputs for the transforms that follow (lines `11 - 14`).

2. Generates a binary image applying 2 color filters (white and yellow) to the HLS image (line `16`) on all 3 channels at the same time.

3. Generates two more binary images applying [Laplacian](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html#laplacian-derivatives) to the grayscale and HLS S-channel verions of the image (lines `18 - 19`) and blurs them to filer out some noise (lines `21 - 22`).

4. Next, all 3 binary images are combined using different thresholds: `(hls_yw_filter == 1) | (laplacianSBlur >= 0.75) | (laplacianBlur >= 0.75)` (lines `24 - 25`).

5. A region of interest filter is applied to filter out the top half of the image approximately, the hood of the car and a small portion of the sides of the road.

The output of the 4th step looks like this:

![Binary Road][image3]

Detailed examples of each step with a wide range of images can be found in [`src/notebooks/Image Preprocessing and Perspective Transformations.ipynb`](src/notebooks/Image%20Preprocessing%20and%20Perspective%20Transformations.ipynb), where multiple options for each step have been considered, tested and calibrated until the result explained above was obtained.

It's worth mentioning that due to time constraints, some options were considered but not tested enough to be able to achieve good results with them, so they were discarded and not used in the final pipeline, even though they could probably help improve the current implementation, which is quite noisy. Some of these features are :

- Histogram equalization (code removed).
- Contrast augmentation (`src/helpers/imageProcessing.py:34 - 42`, used in the 2nd section of the Notebook).
- Alternative color spaces (`CIELAB`, in the 2nd section of the Notebook).
- X and Y sobels, sobel magnitude and sobel direction have been implemented in `src/helpers/imageProcessing.py:144 - 180` and have been compared with Laplacian in the 4th section of the Notebook, but Laplacian did a better job than those other options without too much tunning.


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Again, the perpective transform has been used in `src/helpers/imagePreprocessingPipeline.py:37` as part of the image-processing pipeline and in the last step of [`src/notebooks/Image Preprocessing and Perspective Transformations.ipynb`](src/notebooks/Image%20Preprocessing%20and%20Perspective%20Transformations.ipynb) to provide some examples and visualizations of what it is actually doing:

![Warped Road][image4]

In this last section you can also see how the source `src` and destination `dst` points that are used to generate the transform matrix `M` (and its inverse) are calculated based on the dimensions of the image. This matrix is only computed once in this Notebook and another one in [`src/notebooks/Video Processing.ipynb`](src/notebooks/Video%20Processing.ipynb) and is later reused all the times a perpective transformation needs to be performed.

Both the matrix calculation (`getM`) and the perpective transform (`warper`) have been implemented in [`src/helpers/cameraCalibration.py`](src/helpers/cameraCalibration.py) and are just wrappers around `cv2`'s functionality.


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This has been implemented in a second `pipeline` function in [`src/helpers/polyFitPipeline.py:44 - 109`](src/helpers/polyFitPipeline.py) and multiple examples can be found in [`src/notebooks/Lane Detection (Polynomial).ipynb`](src/notebooks/Lane Detection (Polynomial).ipynb).

The actual implementation is in [`src/helpers/laneFinder.py`](src/helpers/laneFinder.py), in the functions `getFirstTime` and `getFromRegion`.

The first one uses a sliding window to decide which pixels belong to the lane line and uses peaks in the right and left half of the histogram of a especific region of the image (bottom 1/N portion, where N can be adjusted manually) to determine the initial location for the windows.

In the following images we can see the original image, the warped binary with the histogram on top, the sliding window search with the pixels that are assigned to each of the lane lines and the final result of the identified lane:

![Fitting a Polynomial From Histogram Peaks][image5]

The second one uses the polynomial fitted to the previous image and a margin to decide which pixels belong to the each lane line, assuming lane lines in consecutive images should be quite similar:

![Fitting a Polynomial From Previous One][image6]

Once the pixels for each lane line have been selected, a second order polynomial is fitted using `np.polyfit(Y, X, 2)`


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Both calculations are implemented in [`src/helpers/laneFinder.py`](src/helpers/laneFinder.py).

The radius is calculated by the `getRadius` function, that given the lane lines' pixels coordinates, it will convert them to meters using a conversion rate that has been calculated based on the dimensions of the warped image and estimations of the lane's dimensions, and fit a new polynomial to them. Using this new polynomial, it can calculate its curvature, as explained [here](https://www.intmath.com/applications-differentiation/8-radius-curvature.php).

The deviation of the vehicle with respect to the center is calculated in the `getDistanceFromCenter` function, that given the two polynomials, one for each of the two lane lines, it will calculate their value at the bottom of the image, their average (middle point) and the deviation of this point with respect to the center of the image (half its width).


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Again, in [`src/helpers/laneFinder.py:198 - 239`](src/helpers/laneFinder.py), there are 3 functions, `drawOverlay`, `drawOverlayLane` and `drawOverlayInfo`, that will plot the lane back onto the original image, using an inverse perpective transformation, and will also overlay the radius of curvature of the road and the deviation of the vehicle with respect to the center.

The final output looks like this:

![Example Output][image0]


### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](output/videos/004%20-%20Advanced%20Project.mp4)


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are many things that could be improved in this project, but due to time constraints it was not possible.

The various methods that have been tested and discarded to preprocess/filter the images could probably be helpful after investing a bit more time to find proper parameters and combinations for them. The ones that are currently used could also be adjusted better to produce less noise.

Also, their params could be adjusted dinamically based on some properties of the image or conditions. For example, the color filters could be adjusted depending on the brightness of the image.

Regarding the sliding window method that was used, it has some limitations:

- The fixed size of the window makes it work poorly on really curvy roads where the lane lines can look almost horitzontal, as the windows don't move fast enough to keep up with the line's inclination. Dinamically setting their size could be an option.
- The peaks in the histogram used to determine the starting position for the windows assume there's one lane line on each half of the image, but in really curvy roads both lane lines could be on the same side.

  A better approach would be to look for either pairs of peaks that are separated by approximately the width of the lane from each other or single peaks that are separated by approximately the width of the lane from the border of the image (in really sharp turns, the inner lane line might not appear in the image).
  
  Also, the same peaks should appear in multiple horitzontal portions of the image if they actually belong to the lane lines. This could be used to make the detection more robust, filter out invalid peaks that do not belong to a lane line or even detect an arbiratry number of lanes.
  
Lastly, regarding the Line class, while it already helps getting a smoother result in the videos, its implementation is quite basic and could be improved a lot. Also, the current implementation will fail to properly fit a polynomial, even if the lane lines are clearly visible and properly filtered, in consecutive turns, where a higher order polynomial should be used instead.

Moreover, the project's code is currently quite messy and could be better organised, simplified, DRYed and improved in general to make it more performant, robust and maintainable, apart from the possible improvements in the current implementations.


