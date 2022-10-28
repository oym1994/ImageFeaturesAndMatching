#pragma once

#ifdef USE_TENSORRT
#include "tensorrt_generic.h"
#include <ATen/ATen.h>
#include <Eigen/Dense>
#include <torch/csrc/api/include/torch/types.h>
#include <torch/csrc/autograd/variable.h>

#define NETVLAD_DESC_LEN 4096
class NetVLADTensorRT : public TensorRTInferenceGeneric {
public:
  bool enable_perf;

  NetVLADTensorRT(std::string engine_path, int _width, int _height,
                  bool _enable_perf = false);

  void detectFeature(const cv::Mat &image, torch::Tensor &descriptor,
                     torch::Device device = torch::kCUDA);

  virtual std::unordered_map<std::string, std::vector<std::vector<int>>>
  getDynamicInputSizes() {}

  virtual ~NetVLADTensorRT() {
    std::cout << "Deconstruct the NetVLADTensorRT!" << std::endl;
    m_Engine->destroy();
    m_Context->destroy();
    // if (m_Engine) delete m_Engine;
    // m_Engine = nullptr;
    // if (m_Context) delete m_Context;
    // m_Context = nullptr;
  }
};
#endif