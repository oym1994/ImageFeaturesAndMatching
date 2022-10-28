#pragma once

#include "tensorrt_generic.h"
#include <ATen/ATen.h>
#include <opencv2/features2d.hpp>
#include <torch/csrc/api/include/torch/types.h>
#include <torch/csrc/autograd/variable.h>

class SuperGlueTensorRT : public TensorRTInferenceGeneric,
                          public cv::DescriptorMatcher {
  const long unsigned int min_num_, opt_num_, max_num_;

public:
  bool enable_perf;

  SuperGlueTensorRT(std::string engine_path, bool _enable_perf = false,
                    bool dynamic_axes = true, long unsigned int min_num = 50,
                    long unsigned int opt_num = 250,
                    long unsigned int max_num = 500);

  bool match(std::vector<cv::Mat> &data,
             CV_OUT std::vector<cv::DMatch> &matches,
             const cv::Size &image0_size, const cv::Size &image1_size);

  CV_WRAP bool isMaskSupported() const { return true; }

  cv::Ptr<cv::DescriptorMatcher> clone(bool emptyTrainData = false) const {
    cv::Ptr<SuperGlueTensorRT> matcher; //(this); TODO  fix here
    return matcher;
  }

  virtual void knnMatchImpl(cv::InputArray queryDescriptors,
                            std::vector<std::vector<cv::DMatch>> &matches,
                            int k, cv::InputArrayOfArrays masks = cv::noArray(),
                            bool compactResult = false) {}

  virtual void radiusMatchImpl(cv::InputArray queryDescriptors,
                               std::vector<std::vector<cv::DMatch>> &matches,
                               float maxDistance,
                               cv::InputArrayOfArrays masks = cv::noArray(),
                               bool compactResult = false) {}

  static cv::Ptr<cv::DescriptorMatcher>
  create(std::string engine_path, bool _enable_perf = false,
         bool dynamic_axes = true, long unsigned int min_num = 50,
         long unsigned int opt_num = 250, long unsigned int max_num = 500);

  virtual CV_WRAP void match(cv::InputArray queryDescriptors,
                     cv::InputArray trainDescriptors,
                     CV_OUT std::vector<cv::DMatch> &matches,
                     cv::InputArray mask = cv::noArray()) const;

  virtual ~SuperGlueTensorRT() {
    std::cout << "Deconstruct the SuperGlueTensorRT!" << std::endl;
    m_Engine->destroy();
    m_Context->destroy();
    // if (m_Engine) delete m_Engine;
    // m_Engine = nullptr;
    // if (m_Context) delete m_Context;
    // m_Context = nullptr;
  }

  virtual std::unordered_map<std::string, std::vector<std::vector<int>>>
  getDynamicInputSizes();
};
