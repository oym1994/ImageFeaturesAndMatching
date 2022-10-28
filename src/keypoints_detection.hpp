#pragma once
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "contour_extractor.hpp"

#ifdef USE_DEEP_FEATURES

#include "tensorRT/superglue_tensorrt.h"
#include "tensorRT/superpoint_tensorrt.h"
#include <torch/script.h>
#endif

class featureDescriptorFactory
{

public:
  cv::Ptr<cv::DescriptorExtractor> createExtractor(const std::string type, bool xfeature2d)
  {
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (type.compare("BRISK") == 0)
    {

      int threshold = 30;        // FAST/AGAST detection threshold score.
      int octaves = 3;           // detection octaves (use 0 to do single scale)
      float patternScale = 1.0f; // apply this scale to the pattern used for
                                 // sampling the neighbourhood of a keypoint.

      extractor = cv::BRISK::create(threshold, octaves, patternScale);
      xfeature2d = false;
    }
    else if (type.compare("BRIEF") == 0)
    {
      extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
      xfeature2d = true;
    }
    else if (type.compare("ORB") == 0)
    {
      extractor = cv::ORB::create();
      xfeature2d = false;
    }
    else if (type.compare("FREAK") == 0)
    {
      extractor = cv::xfeatures2d::FREAK::create();
      xfeature2d = true;
    }
    else if (type.compare("AKAZE") == 0)
    {
      extractor = cv::AKAZE::create();
      xfeature2d = false;
    }
    else if (type.compare("SIFT") == 0)
    {
      extractor = cv::xfeatures2d::SIFT::create();
      xfeature2d = true;
    }
    else if (type.compare("SURF") == 0)
    {
      extractor = cv::xfeatures2d::SURF::create();
      xfeature2d = true;
    }
#ifdef USE_DEEP_FEATURES
    else if (type.compare("SuperPoint") == 0)
    {
      extractor = SuperPointTensorRT::create();
      xfeature2d = false;
    }
#endif
    else if (type.compare("CONTOUR") == 0)
    {
      extractor = ContourEtractor::create();
      xfeature2d = false;
    }
    else
    {
      std::cerr << "Invalid type: " << type << std::endl;
      abort();
    }
    return extractor;
  }
};

class matcherFactory
{

public:
  cv::Ptr<cv::DescriptorMatcher> createMatcher(const std::string type)
  {
    if (type.compare("BF_L2") == 0)
      return cv::BFMatcher::create(cv::NORM_L2, true);
    else if (type.compare("BF_HAMMING") == 0)
      return cv::BFMatcher::create(cv::NORM_HAMMING, true);
#ifdef USE_DEEP_FEATURES
    else if (type.compare("SuperGlue") == 0)
      return SuperGlueTensorRT::create("../models/SuperGlue.onnx");
#endif
    else if (type.compare("FLANN") == 0)
      return cv::FlannBasedMatcher::create();

    else
    {
      std::cerr << "invalid matcher type!" << std::endl;
      abort();
    }
  }
};

class FeatureDetectorAndMatcher
{

  cv::Ptr<cv::DescriptorExtractor> keypoints_extractor_;

  cv::Ptr<cv::DescriptorExtractor> descriptor_calculator_;

  cv::Ptr<cv::DescriptorMatcher> keypoints_matcher_;

  bool xfeature2d_;

  // desc: BRIEF, ORB, FREAK, AKAZE, SIFT, BRISK, SURF, SuperPoint
  // matcher: BF_HAMMING, BF_L2, SuperGlue, FLANN
  std::unordered_map<std::string, std::unordered_set<std::string>> matcher_desc_{{"FLANN", {"FREAK", "AKAZE", "SIFT", "SURF", "SuperPoint", "ORB", "BRIEF", "BRISK"}},
                                                                                 {"BF_HAMMING", {"ORB", "BRIEF", "BRISK"}},
                                                                                 {"BF_L2", {"FREAK", "AKAZE", "SIFT", "SURF", "SuperPoint"}},
                                                                                 {"SuperGlue", {"SIFT", "SuperPoint"}}};

public:
  FeatureDetectorAndMatcher(
      const cv::Ptr<cv::DescriptorExtractor> &keypoints_extractor,
      const cv::Ptr<cv::DescriptorExtractor> &descriptor_calculator,
      const cv::Ptr<cv::DescriptorMatcher> &matcher)
      : keypoints_extractor_(keypoints_extractor),
        descriptor_calculator_(descriptor_calculator),
        keypoints_matcher_(matcher)
  {
    std::cout << "Construct a feature detector!" << std::endl;
  }

  FeatureDetectorAndMatcher(const std::string feature_type, const std::string descriptor_type, const std::string match_type)
  {
    // check whether the combination of descriptor_type and match_type is valid
    if (checkCombinationFailed(descriptor_type, match_type))
    {
      std::cerr << "Invalid combination of " << descriptor_type << " and " << match_type << std::endl;
      abort();
    }

    // create the feature and descriptor extractor
    featureDescriptorFactory factory;
    keypoints_extractor_ = factory.createExtractor(feature_type, xfeature2d_);
    descriptor_calculator_ = feature_type == descriptor_type ? keypoints_extractor_ : factory.createExtractor(descriptor_type, xfeature2d_);

    matcherFactory match_factory;
    keypoints_matcher_ = match_factory.createMatcher(match_type);
  }

  bool checkCombinationFailed(const std::string descriptor_type, const std::string match_type)
  {
    if (!matcher_desc_.count(match_type))
    {
      std::cerr << "unsupported match type: " << match_type << std::endl;
      return true;
    }
    else if (!matcher_desc_.at(match_type).count(descriptor_type))
    {
      std::cerr << "unsupported descriptor_type " << descriptor_type << " for match type " << match_type << std::endl;
      return true;
    }
    return false;
  }

  void setKeypointExtractor(
      const cv::Ptr<cv::DescriptorExtractor> &keypoints_extractor)
  {
    keypoints_extractor_ = keypoints_extractor;
  }

  void setMatcher(const cv::Ptr<cv::DescriptorMatcher> &keypoints_matcher)
  {
    keypoints_matcher_ = keypoints_matcher;
  }

  void setDescriptorCalculator(
      const cv::Ptr<cv::DescriptorExtractor> &descriptor_calculator)
  {
    descriptor_calculator_ = descriptor_calculator;
  }

  ~FeatureDetectorAndMatcher()
  {
    std::cout << "Deconstruct a feature detector!" << std::endl;
  }

  void detect(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints)
  {
    keypoints_extractor_->detect(image, keypoints);
  }

  void calculateDescriptor(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
  {
    descriptor_calculator_->compute(image, keypoints, descriptors);
  }

  void detectorAndCompute(const cv::Mat &image,
                          std::vector<cv::KeyPoint> &keypoints,
                          cv::Mat &descriptors)
  {

    if (keypoints_extractor_ == descriptor_calculator_ && !xfeature2d_)
    {

      keypoints_extractor_->detectAndCompute(image, cv::noArray(), keypoints,
                                             descriptors);
    }
    else
    {
      detect(image, keypoints);
      calculateDescriptor(image, keypoints, descriptors);
    }
  }

  void matchFeatures(const cv::Mat &desc_source, const cv::Mat &desc_ref,
                     std::vector<cv::DMatch> &matches,
                     std::string selectorType)
  {

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
      keypoints_matcher_->match(
          desc_source, desc_ref,
          matches); // Finds the best match for each descriptor in desc_source
    }
    else if (selectorType.compare("SEL_KNN") ==
             0)
    { // k nearest neighbors (k=2)

      std::vector<std::vector<cv::DMatch>> knn_matches;
      keypoints_matcher_->knnMatch(desc_source, desc_ref, knn_matches, 2);
      double minDescDistRatio = 0.8;
      for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
      {
        if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
        {
          matches.push_back((*it)[0]);
        }
      }
    }
  }
};