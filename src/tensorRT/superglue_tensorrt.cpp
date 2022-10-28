#include "tensorRT/superglue_tensorrt.h"
#include "ATen/Parallel.h"
#include <ATen/Functions.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <vector>

using namespace nvinfer1;

SuperGlueTensorRT::SuperGlueTensorRT(std::string engine_path, bool _enable_perf,
                                     bool dynamic_axes,
                                     long unsigned int min_num,
                                     long unsigned int opt_num,
                                     long unsigned int max_num)
    : TensorRTInferenceGeneric(
          std::vector<std::string>{"descriptors0", "descriptors1", "keypoints0",
                                   "keypoints1", "scores0", "scores1"},
          0, 0,
          std::vector<uint64_t>{256 * max_num, 256 * max_num, 2 * max_num,
                                2 * max_num, max_num, max_num}),
      cv::DescriptorMatcher(), enable_perf(_enable_perf), min_num_(min_num),
      opt_num_(opt_num), max_num_(max_num) {
  assert((min_num_ < opt_num_ && opt_num_ < max_num_) &&
         "Invalid min_num, opt_num and max_num");
  at::set_num_threads(1);
  TensorInfo outputTensorMatchingScore;
  outputTensorMatchingScore.blobName = "matching_score";
  outputTensorMatchingScore.volume = (max_num + 1) * (max_num + 1);

  m_OutputTensors.push_back(outputTensorMatchingScore);

  std::cout << "Trying to init TRT engine of SuperGlue TensorRT" << engine_path
            << std::endl;
  dynamic_axes_ = dynamic_axes;
  init(engine_path);
}

std::unordered_map<std::string, std::vector<std::vector<int>>>
SuperGlueTensorRT::getDynamicInputSizes() {
  std::unordered_map<std::string, std::vector<std::vector<int>>>
      dynamic_inputs_sizes{
          {"descriptors0",
           std::vector<std::vector<int>>{std::vector<int>{1, 256, min_num_},
                                         std::vector<int>{1, 256, opt_num_},
                                         std::vector<int>{1, 256, max_num_}}},
          {"descriptors1",
           std::vector<std::vector<int>>{std::vector<int>{1, 256, min_num_},
                                         std::vector<int>{1, 256, opt_num_},
                                         std::vector<int>{1, 256, max_num_}}},
          {"keypoints0",
           std::vector<std::vector<int>>{std::vector<int>{1, min_num_, 2},
                                         std::vector<int>{1, opt_num_, 2},
                                         std::vector<int>{1, max_num_, 2}}},
          {"keypoints1",
           std::vector<std::vector<int>>{std::vector<int>{1, min_num_, 2},
                                         std::vector<int>{1, opt_num_, 2},
                                         std::vector<int>{1, max_num_, 2}}},
          {"scores0",
           std::vector<std::vector<int>>{std::vector<int>{1, min_num_},
                                         std::vector<int>{1, opt_num_},
                                         std::vector<int>{1, max_num_}}},
          {"scores1",
           std::vector<std::vector<int>>{std::vector<int>{1, min_num_},
                                         std::vector<int>{1, opt_num_},
                                         std::vector<int>{1, max_num_}}}};

  return dynamic_inputs_sizes;
}

void SuperGlueTensorRT::match(cv::InputArray queryDescriptors,
                              cv::InputArray trainDescriptors,
                              CV_OUT std::vector<cv::DMatch> &matches,
                              cv::InputArray mask) const {

  TicAndToc tic_total;
  std::cout << "SuperGlueTensorRT::match " << std::endl;

  cv::Mat mat0 = queryDescriptors.getMat(), mat1 = trainDescriptors.getMat();

  cv::Mat desc0 = mat0.colRange(0, mat0.cols - 3),
          desc1 = mat1.colRange(0, mat1.cols - 3);
  std::cout << "desc0 cols: " << desc0.cols << std::endl;

  long unsigned int kpt0_num = desc0.rows, kpt1_num = desc1.rows;

  if (kpt0_num > max_num_ || kpt0_num < min_num_ || kpt1_num > max_num_ ||
      kpt1_num < min_num_) {
    std::cout << "Too few or too many features!" << std::endl;
    return;
  }

  cv::Mat kpts0 = mat0.colRange(mat0.cols - 3, mat0.cols - 1),
          kpts1 = mat1.colRange(mat1.cols - 3, mat1.cols - 1);
  std::cout << "kpts0 cols: " << kpts0.cols << std::endl;

  cv::Mat scores0 = mat0.colRange(mat0.cols - 1, mat0.cols),
          scores1 = mat1.colRange(mat1.cols - 1, mat1.cols);

  if (dynamic_axes_) {
    // set current inputs sizes
    // m_OutputTensors[0].volume = (kpt0_num + 1) * (kpt1_num + 1);

    std::unordered_map<std::string, std::vector<int>> current_inputs_sizes{
        {"descriptors0", std::vector<int>{1, 256, kpt0_num}},
        {"descriptors1", std::vector<int>{1, 256, kpt1_num}},
        {"keypoints0", std::vector<int>{1, kpt0_num, 2}},
        {"keypoints1", std::vector<int>{1, kpt1_num, 2}},
        {"scores0", std::vector<int>{1, kpt0_num}},
        {"scores1", std::vector<int>{1, kpt1_num}}};
    assert((kpt0_num >= min_num_ && kpt0_num <= max_num_) &&
           "Invalid kpt0_num");
    assert((kpt1_num >= min_num_ && kpt1_num <= max_num_) &&
           "Invalid kpt1_num");

    for (unsigned int i = 0; i < m_InputBindingIndexGroup.size(); i++) {
      std::vector<int> dims_vec = current_inputs_sizes.at(
          m_InputBlobNameGroup[m_InputBindingIndexGroup[i]]);
      auto dims = m_Engine->getBindingDimensions(m_InputBindingIndexGroup[i]);
      switch (dims_vec.size()) {
      case 2:
        m_Context->setBindingDimensions(i, Dims2(dims_vec[0], dims_vec[1]));
        break;
      case 3:
        m_Context->setBindingDimensions(
            i, Dims3(dims_vec[0], dims_vec[1], dims_vec[2]));
        break;
      default:
        break;
      }
    }
  }

  TicAndToc tic;

  std::cout << "kpts0: \n" << kpts0 << std::endl;

  doInference(std::vector<cv::Mat>{desc0.t(), desc1.t(), kpts0, kpts1,
                                   scores0.t(), scores1.t()});

  if (enable_perf) {
    std::cout << "SuperGlue Inference Time(ms): " << tic.toc() << std::endl;
  }

  total_call_times_++;
  total_time_cost_ += tic.toc();

  const float match_threshold = 0.15; // reference from the paper

  auto options = torch::TensorOptions().dtype(torch::kFloat32);

  auto score_matrix = at::from_blob(m_OutputTensors[0].hostBuffer,
                                    {1, kpt0_num + 1, kpt1_num + 1}, options)
                          .squeeze(0);

  auto valid_score_matrix =
      score_matrix.narrow(0, 0, kpt0_num).narrow(1, 0, kpt1_num);
  std::cout << "valid_score_matrix: " << valid_score_matrix << std::endl;

  std::tuple<torch::Tensor, torch::Tensor> max0 = (torch::max)(
                                               valid_score_matrix, 1),
                                           max1 = (torch::max)(
                                               valid_score_matrix, 0);

  torch::Tensor indices0 = std::get<1>(max0),
                indices1 = std::get<1>(max1); // 1 * 500

  auto arange_like0 = indices0.new_full(indices0.size(0), 1).cumsum(0) - 1,
       arange_like1 = indices1.new_full(indices1.size(0), 1).cumsum(0) - 1;

  auto mutual0 = (arange_like0 == indices1.gather(0, indices0));

  auto mutual1 = (arange_like1 == indices0.gather(0, indices1));

  auto zero0 = mutual0.new_zeros(mutual0.sizes(), torch::kFloat32);

  auto mscores0 = torch::where(
      mutual0, std::get<0>(max0).exp(),
      zero0); // torch::where(condition，a，b): if condition：choose to select
              // a，or choose b as the output

  auto zero1 = mutual1.new_zeros(mutual1.sizes(), torch::kFloat32);

  auto mscores1 = torch::where(mutual1, mscores0.gather(0, indices1), zero1);

  auto valid0 = mutual0 & (mscores0 > match_threshold);
  auto valid1 = mutual1 & valid0.gather(0, indices1);

  indices0 =
      torch::where(valid0, indices0, indices0.new_zeros(mutual0.sizes()) - 1);
  indices1 =
      torch::where(valid1, indices1, indices1.new_zeros(indices1.sizes()) - 1);

  torch::Tensor matches_tensor = indices0;

  torch::Tensor valid = at::nonzero(matches_tensor > 0).squeeze();

  torch::Tensor mkpts_query = valid; // matched index
  torch::Tensor mkpts_ref = matches_tensor.index_select(0, valid);

  torch::Tensor matching_scores0 = mscores0.unsqueeze(0);

  int n = valid.size(0);
  std::cout << "n: " << n << std::endl;
  matches.reserve(n);

  for (int i = 0; i < n; ++i) {
    const int query_kpt_index = mkpts_query[i].item<int>();
    const int ref_kpt_index = mkpts_ref[i].item<int>();
    matches.push_back(cv::DMatch(query_kpt_index, ref_kpt_index, 0.5));
  }

  if (enable_perf) {
    std::cout << "SuperGlue total Time(ms): " << tic_total.toc() << std::endl;
  }
}

cv::Ptr<cv::DescriptorMatcher>
SuperGlueTensorRT::create(std::string engine_path, bool enable_perf,
                          bool dynamic_axes, long unsigned int min_num,
                          long unsigned int opt_num,
                          long unsigned int max_num) {

  return new SuperGlueTensorRT(engine_path, enable_perf, dynamic_axes, min_num,
                               opt_num, max_num);
}

bool SuperGlueTensorRT::match(std::vector<cv::Mat> &data,
                              CV_OUT std::vector<cv::DMatch> &matches,
                              const cv::Size &image0_size,
                              const cv::Size &image1_size) {
  TicAndToc tic_total;

  long unsigned int kpt0_num = data[0].cols, kpt1_num = data[1].cols;

  if (kpt0_num > max_num_ || kpt0_num < min_num_ || kpt1_num > max_num_ ||
      kpt1_num < min_num_) {
    std::cout << "Too few or too many features!" << std::endl;
    return false;
  }

  std::cout << data[0] << std::endl;
  std::cout << "~~~~~~~~~~~~~~" << std::endl;
  std::cout << data[1] << std::endl;
  std::cout << "~~~~~~~~~~~~~~" << std::endl;
  std::cout << data[2] << std::endl;
  std::cout << "~~~~~~~~~~~~~~" << std::endl;
  std::cout << data[3] << std::endl;
  std::cout << "~~~~~~~~~~~~~~" << std::endl;
  std::cout << data[4] << std::endl;
  std::cout << "~~~~~~~~~~~~~~" << std::endl;
  std::cout << data[5] << std::endl;

  {
    cv::Mat &kpts0 = data[2];
    cv::Mat &kpts1 = data[3];

    // preprocess: normalize keypoints, reference fro superglue network
    cv::Mat center1 = kpts1.clone();
    center1 = image1_size.width * 0.5;
    center1.col(1).rowRange(0, kpts1.rows) = image1_size.height * 0.5;
    kpts1 = (kpts1 - center1) / (image1_size.width * 0.7);

    cv::Mat center0 = kpts0.clone();
    center0 = image0_size.width * 0.5;
    center0.col(1).rowRange(0, kpts0.rows) = image0_size.height * 0.5;
    kpts0 = (kpts0 - center0) / (image0_size.width * 0.7);
  }

  if (dynamic_axes_) {
    // set current inputs sizes
    m_OutputTensors[0].volume = (kpt0_num + 1) * (kpt1_num + 1);

    std::unordered_map<std::string, std::vector<int>> current_inputs_sizes{
        {"descriptors0", std::vector<int>{1, 256, kpt0_num}},
        {"descriptors1", std::vector<int>{1, 256, kpt1_num}},
        {"keypoints0", std::vector<int>{1, kpt0_num, 2}},
        {"keypoints1", std::vector<int>{1, kpt1_num, 2}},
        {"scores0", std::vector<int>{1, kpt0_num}},
        {"scores1", std::vector<int>{1, kpt1_num}}};
    assert((kpt0_num >= min_num_ && kpt0_num <= max_num_) &&
           "Invalid kpt0_num");
    assert((kpt1_num >= min_num_ && kpt1_num <= max_num_) &&
           "Invalid kpt1_num");

    for (unsigned int i = 0; i < m_InputBindingIndexGroup.size(); i++) {
      std::vector<int> dims_vec = current_inputs_sizes.at(
          m_InputBlobNameGroup[m_InputBindingIndexGroup[i]]);
      auto dims = m_Engine->getBindingDimensions(m_InputBindingIndexGroup[i]);
      switch (dims_vec.size()) {
      case 2:
        m_Context->setBindingDimensions(i, Dims2(dims_vec[0], dims_vec[1]));
        break;
      case 3:
        m_Context->setBindingDimensions(
            i, Dims3(dims_vec[0], dims_vec[1], dims_vec[2]));
        break;
      default:
        break;
      }
    }
  }

  TicAndToc tic;

  doInference(data);

  if (enable_perf) {
    std::cout << "SuperGlue Inference Time(ms): " << tic.toc() << std::endl;
  }

  total_call_times_++;
  total_time_cost_ += tic.toc();

  const float match_threshold = 0.15; // reference from the paper

  auto options = torch::TensorOptions().dtype(torch::kFloat32);

  auto score_matrix = at::from_blob(m_OutputTensors[0].hostBuffer,
                                    {1, kpt0_num + 1, kpt1_num + 1}, options)
                          .squeeze(0);

  auto valid_score_matrix =
      score_matrix.narrow(0, 0, kpt0_num).narrow(1, 0, kpt1_num);

  std::cout << "valid_score_matrix \n" << valid_score_matrix << std::endl;

  std::tuple<torch::Tensor, torch::Tensor> max0 = (torch::max)(
                                               valid_score_matrix, 1),
                                           max1 = (torch::max)(
                                               valid_score_matrix, 0);

  torch::Tensor indices0 = std::get<1>(max0),
                indices1 = std::get<1>(max1); // 1 * 500

  auto arange_like0 = indices0.new_full(indices0.size(0), 1).cumsum(0) - 1,
       arange_like1 = indices1.new_full(indices1.size(0), 1).cumsum(0) - 1;

  auto mutual0 = (arange_like0 == indices1.gather(0, indices0));

  auto mutual1 = (arange_like1 == indices0.gather(0, indices1));

  auto zero0 = mutual0.new_zeros(mutual0.sizes(), torch::kFloat32);

  auto mscores0 = torch::where(
      mutual0, std::get<0>(max0).exp(),
      zero0); // torch::where(condition，a，b): if condition：choose to select
              // a，or choose b as the output

  auto zero1 = mutual1.new_zeros(mutual1.sizes(), torch::kFloat32);

  auto mscores1 = torch::where(mutual1, mscores0.gather(0, indices1), zero1);

  auto valid0 = mutual0 & (mscores0 > match_threshold);
  auto valid1 = mutual1 & valid0.gather(0, indices1);

  indices0 =
      torch::where(valid0, indices0, indices0.new_zeros(mutual0.sizes()) - 1);
  indices1 =
      torch::where(valid1, indices1, indices1.new_zeros(indices1.sizes()) - 1);

  torch::Tensor matches0 = indices0;
  torch::Tensor matching_scores0 = mscores0.unsqueeze(0);

  torch::Tensor matches_tensor = indices0;

  torch::Tensor valid = at::nonzero(matches_tensor > 0).squeeze();

  torch::Tensor mkpts_query = valid; // matched index
  torch::Tensor mkpts_ref = matches_tensor.index_select(0, valid);

  int n = valid.size(0);
  std::cout << "n: " << n << std::endl;
  matches.reserve(n);

  for (int i = 0; i < n; ++i) {
    const int query_kpt_index = mkpts_query[i].item<int>();
    const int ref_kpt_index = mkpts_ref[i].item<int>();
    matches.push_back(cv::DMatch(query_kpt_index, ref_kpt_index, 0.5));
  }

  if (enable_perf) {
    std::cout << "SuperGlue total Time(ms): " << tic_total.toc() << std::endl;
  }

  return true;
}