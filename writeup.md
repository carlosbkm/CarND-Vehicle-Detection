##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[carnotcar]: ./writeup_images/car-notcar.png
[hog]: ./writeup_images/example-hog.png
[bboxes]: ./writeup_images/example-boxes.png
[bboxes2]: ./writeup_images/no-false-positive.png
[heat]: ./writeup_images/heat-map.png
[band1]: ./writeup_images/band1.png
[band2]: ./writeup_images/band2.png
[band3]: ./writeup_images/band3.png
[image]: ./writeup_images/example-image.png
[3dhist]: ./writeup_images/3dhistograms.png
[histograms]: ./writeup_images/histograms.png
[norm]: ./writeup_images/normalized-features.png
[video1]: ./project_video.mp4
[pipe1]: ./writeup_images/pipe1.png
[pipe2]: ./writeup_images/pipe1.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][carnotcar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog]


####2. Explain how you settled on your final choice of HOG parameters.

First I experimented with various color spaces

![alt text][3dhist]

Then I played with different parameters, and switching between channels to see the difference. Finally I found that combining all the channels with color space 'YCrCb' yield the best results for me. I also experimented with the spatial bin dimensions and many others, and finnally sticked to these values:
```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()
x_start_stop=[None, None]
xy_window=(96, 96)
xy_overlap=(0.5, 0.5)
```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using with the following parameters:
```python
svc = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=2)
```
This can be found in the 10th cell of the [project notebook][./project.ipynb]. I also tried a SVC classifier but since I didn't find any improvement, I kept LinearSVC.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented the sliding window search in the method `find_cars`of the 13th cell. I chose 3 different window heights based on the size of the car as they move to the horizon. Also, I played with different  window scales and found that 1.25 and 1.5 were performing good.

These are the params I used:
```python
first_y_limit = [390, 510]
second_y_limit = [390, 610]
third_y_limit = [490, 660]
another_y = [400, 656]
x_limits = [0, 1280]
windows_params = {'sizes':[(64, 64),(96,96),(128,128), (256, 256)],
               'y_limits': [first_y_limit,second_y_limit,third_y_limit],
               'x_limits': [x_limits,x_limits,x_limits, x_limits],
                 'scales':[1.25, 1.5]}
```
Which gave this result:
![alt text][bboxes]

And these are the three different scan bands according to the window sizes:
Band 1
![alt text][band1]

Band 2
![alt text][band2]

Band 3
![alt text][band3]


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

With scale = 1.2
![alt text][pipe1]

With scale = 2
![alt text][pipe2]



---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. I created the class Heatmap in cell 16th for this purpose.

From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here is an example of the image and the heatmaps:
![alt text][heat]

And here is the final result after applying the threshold to get rid of false positives:
![alt text][bboxes2]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found many difficulties to work with the hog channels, because after applying some transformations I frequently got many errors with the arrays shapes.

Also, I found very hard to tweak the values for the windows to succesfully detect the cars. More research should be done to get better fit rectangles to the images of the vehicles.

Although the LinearSVC produced a very high accuracy, I saw a tremendous improvement in the accuracy of the detection when training the model with more images. During the development I started using the small dataset, I wasn't able to detect the white car properly. But with the same parameters, just by using all the images available from the big dataset, then I could detect the vehicles with no problems. So I guess that for this kind of problems, having a bigger (and balanced) dataset would make a difference.


