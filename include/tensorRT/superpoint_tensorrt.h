#pragma once

#include "tensorrt_generic.h"
// #include <torch/csrc/autograd/variable.h>
#include <ATen/ATen.h>
// #include <Eigen/Dense>
#include <opencv2/features2d.hpp>
#include <torch/csrc/api/include/torch/types.h>

#define SP_DESC_RAW_LEN 256
class SuperPointTensorRT : public TensorRTInferenceGeneric,
                           public cv::DescriptorExtractor {
  // Eigen::MatrixXf pca_comp_T;
  // Eigen::RowVectorXf pca_mean;
  torch::Tensor m_FullDesc;

  int request_feat_num_, enforce_uniformity_radius_;

  float score_thresh_;

public:
  bool enable_perf;

  SuperPointTensorRT(std::string engine_path, uint64_t _width, uint64_t _height,
                     int request_feat_num = 300,
                     int enforce_uniformity_radius = 12,
                     float score_thresh = 0.001, bool _enable_perf = false);

  CV_WRAP static cv::Ptr<cv::DescriptorExtractor> create(int kpts_num = 500,
                                                         float score_thr = 0.1);

  void
  computeDescriptors(/*const torch::Tensor &mProb,*/ const torch::Tensor &desc,
                     const std::vector<cv::KeyPoint> &keypoints,
                     std::vector<float> &local_descriptors) const;

  void computeDescriptors(const std::vector<cv::KeyPoint> &keypoints,
                          std::vector<float> &local_descriptors) const;

  void inference(const cv::Mat &input, std::vector<cv::Point2f> &keypoints,
                 std::vector<float> &local_descriptors);

  void detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint> &keypoints,
              cv::InputArray mask = cv::noArray());

  void detectAndCompute(cv::InputArray image, cv::InputArray mask,
                        CV_OUT std::vector<cv::KeyPoint> &keypoints,
                        cv::OutputArray descriptors,
                        bool useProvidedKeypoints = false);

  void compute(cv::InputArray image,
               CV_OUT CV_IN_OUT std::vector<cv::KeyPoint> &keypoints,
               cv::OutputArray descriptors);

  // void compute(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints,
  // cv::OutputArray descriptors);

  void getKeyPointsScores(const cv::Mat &prob,
                          const cv::Mat &detect_mask, float threshold,
                          std::vector<cv::KeyPoint> &keypoints,
                          int enforce_uniformity_radius, int request_feat_num);

  std::unordered_map<std::string, std::vector<std::vector<int>>>
  getDynamicInputSizes() {}

  ~SuperPointTensorRT() {
    std::cout << "Deconstruct the SuperPoint TensorRT!" << std::endl;
    // m_Engine->destroy();
    // m_Context->destroy();
    // if (m_Engine) delete m_Engine;
    // m_Engine = nullptr;
    // if (m_Context) delete m_Context;
    // m_Context = nullptr;
  }
};
