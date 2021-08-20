# Image_Processing
Simple image processing projects with descriptions.

## Summary
These are two image processing projects that perform multiple image transformations on the given set of images. 
The content of each project is the following:

**Project 1:**
  - Part 1:
    - Laplacian transform of an image
    - Application of Sobel's x and y gradients
    - Sum of Sobel gradients (image sharpening)
    - 5x5 kernel average of Sobel filter (pixel averaging)
    - Addition of averaged image to the sharpened image
    - Multiplication of the pixel averaging with the sharpened image
    - Addition of the previous transformation to the original image in order to make it clearer with adjustable power.
  - Part 2:
    - Application of DFT for images
    - Frequency domain high and low pass filtering
    - Image denoising 
    - Image deblurring

**Project 2:**
  - Part 1:
    - RGB components of image
    - Modification of B component by histogram equalization
  - Part 2:
    - Edge Pixel detection
    - Edge linking by Hough transform
  - Part 3:
    - Otsu's algorithm for image partition
 
__Note: For more information check descriptions__

## Structure
In **Project 1** the 3 files that main contain every operation are: 
 * project_1.py (rest of transformations)
 * project_1_deblurr.py (deblurring)
 * project_1_noise_rem.py (noise removal)

In **Project 2** most of the operations are in: 
 * project_2.py (part 1 and some of part 2)
 * Matlab files (implementations of parts 2 and 3 (incomplete))

## Run Locally
  - Run: git clone https://github.com/jjpro100/Image_Processing.git
  - Compile and run with your local python and matlab compiler

## Versions
  - Python3 or higher
  - Matlab 2017 or higher
