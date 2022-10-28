#include "tensorRT/netvlad_tensorrt.h"
#include <torch/torch.h>
#include <ATen/Functions.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "ATen/Parallel.h"

NetVLADTensorRT::NetVLADTensorRT(std::string engine_path, int _width, int _height, bool _enable_perf) :
    TensorRTInferenceGeneric(std::vector<std::string>{"image"}, _width, _height,
                             std::vector<uint64_t>{_width * _height * 3}),
    enable_perf(_enable_perf)
{
    at::set_num_threads(1);
    TensorInfo outputTensorDesc;
    outputTensorDesc.blobName = "global_descriptor";
    outputTensorDesc.volume   = NETVLAD_DESC_LEN;
    m_OutputTensors.push_back(outputTensorDesc);
    std::cout << "Trying to init TRT engine of NevVLADTensorRT" << engine_path << std::endl;
    init(engine_path);
}

void NetVLADTensorRT::detectFeature(const cv::Mat &input, torch::Tensor &descriptor, torch::Device device)
{
    TicToc tic;
    cv::Mat _input;
    assert(input.rows == height && input.cols == width && "Input image must have same size with network");

    input.convertTo(_input, CV_32F, 1 / 255.0);
    doInference(_input);
    if (enable_perf) {
        std::cout << "Inference Time " << tic.toc();
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32);

    descriptor = at::from_blob(m_OutputTensors[0].hostBuffer, {1, NETVLAD_DESC_LEN}, options).to(device);

    if (enable_perf) {
        std::cout << " get global descriptors: " << tic.toc() << "ms" << std::endl;
    }
}