# Image Features Matching

<img src="images/keypoints.png" width="820" height="248" />

## motivation
Design an abstract class where multiple image features, descriptors and matching methods can be chosen from the config file, including traditional
and deep learning based methods, for some visual feature based tasks, such as visual slam, AR/VR and object tracking.

Supportted image features: BRISK, ORB, AKAZE, SIFT, SURF, CONTOUR, SuperPoint (with TensorRT accelerated) 
Supportted descriptor types: BRIEF, ORB, FREAK, AKAZE, SIFT, BRISK, SURF, SuperPoint (with TensorRT accelerated) 
Supported matching methods: BF_HAMMING, BF_L2, SuperGlue (with TensorRT accelerated) 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* optional for USE_DEEP_FEATURES
  * TensorRT 8.2.3
  * Libtorch 1.8

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./image_matching [keypoints type] [descriptor type]`.

-----
# Reference
[SFND_2D_Feature_Tracking](https://github.com/godloveliang/SFND_2D_Feature_Tracking)

[feature detection](https://github.com/deepanshut041/feature-detection)

[Image Matching from Handcrafted to Deep Features: A Survey](https://www.researchgate.net/publication/343429659_Image_Matching_from_Handcrafted_to_Deep_Features_A_Survey)

[Image Matching Across Wide Baselines: From Paper to Practice](https://www.researchgate.net/publication/339674798_Image_Matching_across_Wide_Baselines_From_Paper_to_Practice)

[Image Registration Techniques: A Survey](https://www.researchgate.net/publication/321342677_Image_Registration_Techniques_A_Survey)

[ORB](https://www.researchgate.net/publication/221111151_ORB_an_efficient_alternative_to_SIFT_or_SURF)

[SIFT](http://web.itu.edu.tr/~aygunme/sift.pdf)

[SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)

[SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)

[BRISK](https://www.researchgate.net/publication/221110715_BRISK_Binary_Robust_invariant_scalable_keypoints)

[BRIEF](https://www.researchgate.net/publication/44198726_BRIEF_Binary_Robust_Independent_Elementary_Features)

[FREAK](https://www.researchgate.net/publication/258848394_FREAK_Fast_retina_keypoint)

[AKAZE](https://www.researchgate.net/publication/257142102_Fast_Explicit_Diffusion_for_Accelerated_Features_in_Nonlinear_Scale_Spaces)
