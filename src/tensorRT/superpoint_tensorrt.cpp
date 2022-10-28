#include "tensorRT/superpoint_tensorrt.h"
#include <ATen/Functions.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <vector>
// #include "loop_defines.h"
#include "ATen/Parallel.h"

// #define USE_PCA
// NMS code is modified from https://github.com/KinglittleQ/SuperPoint_SLAM
void NMS2(std::vector<cv::Point2f> det, cv::Mat conf,
          std::vector<cv::KeyPoint> &pts, int border, int dist_thresh,
          int img_width, int img_height, int max_num);

#define MAXBUFSIZE 100000

SuperPointTensorRT::SuperPointTensorRT(std::string engine_path, uint64_t _width,
                                       uint64_t _height, int request_feat_num,
                                       int enforce_uniformity_radius,
                                       float score_thresh, bool _enable_perf)
    : TensorRTInferenceGeneric(std::vector<std::string>{"image"}, _width,
                               _height,
                               std::vector<uint64_t>{_width * _height}),
      cv::DescriptorExtractor(), request_feat_num_(request_feat_num),
      enforce_uniformity_radius_(enforce_uniformity_radius),
      score_thresh_(score_thresh), enable_perf(_enable_perf) {

  std::cout << "enforce_uniformity_radius: " << enforce_uniformity_radius_
            << std::endl;
  std::cout << "request_feat_num: " << request_feat_num_ << std::endl;

  at::set_num_threads(1);
  TensorInfo outputTensorSemi, outputTensorDesc;
  outputTensorSemi.blobName = "semi";
  outputTensorDesc.blobName = "desc";
  outputTensorSemi.volume = height * width;
  outputTensorDesc.volume = 1 * SP_DESC_RAW_LEN * height / 8 * width / 8;
  // m_InputSize               = height * width;
  m_OutputTensors.push_back(outputTensorSemi);
  m_OutputTensors.push_back(outputTensorDesc);
  std::cout << "Trying to init TRT engine of SuperPointTensorRT" << engine_path
            << std::endl;
  init(engine_path);
}

void SuperPointTensorRT::inference(const cv::Mat &input,
                                   std::vector<cv::Point2f> &keypoints,
                                   std::vector<float> &local_descriptors) {}

cv::Ptr<cv::DescriptorExtractor> SuperPointTensorRT::create(int kpts_num,
                                                            float score_thr) {

  return new SuperPointTensorRT(
      "../models/SuperPoint.onnx", 640, 400);
}

void SuperPointTensorRT::detect(cv::InputArray _image,
                                CV_OUT std::vector<cv::KeyPoint> &keypoints,
                                cv::InputArray _mask) {

  cv::Mat image = _image.getMat(), mask = _mask.getMat();

  TicAndToc tic;
  cv::Mat _input;
  assert(image.rows == height && image.cols == width &&
         "Input image must have same size with network");
  if (image.rows != height || image.cols != width) {
    cv::resize(image, image, cv::Size(width, height));
    image.convertTo(_input, CV_32F, 1 / 255.0);
  } else {
    image.convertTo(_input, CV_32F, 1 / 255.0);
  }
  doInference(_input);

  cv::Mat Prob = cv::Mat(height, width, CV_32F, m_OutputTensors[0].hostBuffer);

  getKeyPointsScores(Prob, mask, score_thresh_, keypoints,
                     enforce_uniformity_radius_, request_feat_num_);
}

void SuperPointTensorRT::compute(cv::InputArray _image,
                                 CV_OUT CV_IN_OUT
                                 std::vector<cv::KeyPoint> &keypoints,
                                 cv::OutputArray descriptors) {

  if (keypoints.empty())
    return;

  cv::Mat image = _image.getMat();
  cv::Mat input;

  assert(image.rows == height && image.cols == width &&
         "Input image must have same size with network");
  if (image.rows != height || image.cols != width) {
    cv::resize(image, image, cv::Size(width, height));
    image.convertTo(input, CV_32F, 1 / 255.0);
  } else {
    image.convertTo(input, CV_32F, 1 / 255.0);
  }
  doInference(input);
  auto mDesc = at::from_blob(m_OutputTensors[1].hostBuffer,
                             {1, SP_DESC_RAW_LEN, height / 8, width / 8},
                             torch::TensorOptions().dtype(torch::kFloat32));

  std::vector<float> local_descriptors;
  computeDescriptors(mDesc, keypoints, local_descriptors);

  int r = keypoints.size();

  int c = local_descriptors.size() / r;
  descriptors.getMatRef() =
      cv::Mat(r, c, CV_32FC1, (float *)local_descriptors.data()).clone();
}

void SuperPointTensorRT::detectAndCompute(
    cv::InputArray _image, cv::InputArray _mask,
    CV_OUT std::vector<cv::KeyPoint> &keypoints, cv::OutputArray descriptors,
    bool useProvidedKeypoints) {

  bool do_keypoints = !useProvidedKeypoints;
  bool do_descriptors = descriptors.needed();

  cv::Mat image = _image.getMat(), mask = _mask.getMat();

  TicAndToc tic;
  cv::Mat _input;
  assert(image.rows == height && image.cols == width &&
         "Input image must have same size with network");
  if (image.rows != height || image.cols != width) {
    cv::resize(image, image, cv::Size(width, height));
    image.convertTo(_input, CV_32F, 1 / 255.0);
  } else {
    image.convertTo(_input, CV_32F, 1 / 255.0);
  }
  doInference(_input);

  cv::Mat Prob = cv::Mat(height, width, CV_32F, m_OutputTensors[0].hostBuffer);

  getKeyPointsScores(Prob, mask, score_thresh_, keypoints,
                     enforce_uniformity_radius_, request_feat_num_);

  std::vector<float> local_descriptors;
  auto mDesc = at::from_blob(m_OutputTensors[1].hostBuffer,
                             {1, SP_DESC_RAW_LEN, height / 8, width / 8},
                             torch::TensorOptions().dtype(torch::kFloat32));
  computeDescriptors(mDesc, keypoints, local_descriptors);

  int r = keypoints.size();

  if (r == 0)
    return;

  int c = local_descriptors.size() / r;
  descriptors.getMatRef() =
      cv::Mat(r, c, CV_32FC1, (float *)local_descriptors.data()).clone();

  total_call_times_++;
  total_time_cost_ += tic.toc();
}

void SuperPointTensorRT::getKeyPointsScores(
    const cv::Mat &prob, const cv::Mat &detect_mask, float threshold,
    std::vector<cv::KeyPoint> &keypoints, int enforce_uniformity_radius,
    int request_feat_num) {
  auto mask = (prob > threshold);

  // cv::imshow("original mask", mask);

  if (!detect_mask.empty())
    mask = mask & detect_mask;

  cv::imshow("mask", mask);

  std::vector<cv::Point> kps;
  cv::findNonZero(mask, kps);
  std::vector<cv::Point2f> keypoints_no_nms;
  for (unsigned int i = 0; i < kps.size(); i++) {
    keypoints_no_nms.push_back(cv::Point2f(kps[i].x, kps[i].y));
  }

  cv::Mat conf(keypoints_no_nms.size(), 1, CV_32F);
  for (size_t i = 0; i < keypoints_no_nms.size(); i++) {
    int x = keypoints_no_nms[i].x;
    int y = keypoints_no_nms[i].y;
    conf.at<float>(i, 0) = prob.at<float>(y, x);
  }

  const int border = enforce_uniformity_radius;

  NMS2(keypoints_no_nms, conf, keypoints, border, enforce_uniformity_radius,
       width, height, request_feat_num);

  for (auto &kp : keypoints) {
    kp.response = prob.at<float>(kp.pt.y, kp.pt.x);
    kp.octave = 0;
  }
}

void SuperPointTensorRT::computeDescriptors(
    const std::vector<cv::KeyPoint> &keypoints,
    std::vector<float> &local_descriptors) const {
  computeDescriptors(m_FullDesc, keypoints, local_descriptors);
}

void SuperPointTensorRT::computeDescriptors(
    const torch::Tensor &mDesc, const std::vector<cv::KeyPoint> &keypoints,
    std::vector<float> &local_descriptors) const {
  cv::Mat kpt_mat(keypoints.size(), 2, CV_32F); // [n_keypoints, 2]  (y, x)
  for (size_t i = 0; i < keypoints.size(); i++) {
    kpt_mat.at<float>(i, 0) = (float)keypoints[i].pt.y;
    kpt_mat.at<float>(i, 1) = (float)keypoints[i].pt.x;
  }

  auto fkpts = torch::from_blob(kpt_mat.data, {(long int)keypoints.size(), 2},
                                torch::kFloat);

  auto grid = torch::zeros({1, 1, fkpts.size(0), 2}); // [1, 1, n_keypoints, 2]
  grid[0][0].slice(1, 0, 1) = 2.0 * fkpts.slice(1, 1, 2) / width - 1;  // x
  grid[0][0].slice(1, 1, 2) = 2.0 * fkpts.slice(1, 0, 1) / height - 1; // y
  auto desc = torch::grid_sampler(mDesc, grid, 0, 0, 0);

  desc = desc.squeeze(0).squeeze(1);

  // normalize to 1
  auto dn = torch::norm(desc, 2, 1);
  desc = desc.div(torch::unsqueeze(dn, 1));

  desc = desc.transpose(0, 1).contiguous();
  desc = desc.to(torch::kCPU);

  local_descriptors = std::vector<float>(desc.data_ptr<float>(),
                                         desc.data_ptr<float>() + desc.numel());
}

bool pt_conf_comp(std::pair<cv::Point2f, double> i1,
                  std::pair<cv::Point2f, double> i2) {
  return (i1.second > i2.second);
}

void NMS2(std::vector<cv::Point2f> det, cv::Mat conf,
          std::vector<cv::KeyPoint> &pts, int border, int dist_thresh,
          int img_width, int img_height, int max_num) {
  std::vector<cv::Point2f> pts_raw = det;

  std::vector<std::pair<cv::Point2f, double>> pts_conf_vec;

  cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
  cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

  cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32FC1);

  grid.setTo(0);
  inds.setTo(0);
  confidence.setTo(0);

  for (unsigned int i = 0; i < pts_raw.size(); i++) {
    int uu = (int)pts_raw[i].x;
    int vv = (int)pts_raw[i].y;

    grid.at<char>(vv, uu) = 1;
    inds.at<unsigned short>(vv, uu) = i;

    confidence.at<float>(vv, uu) = conf.at<float>(i, 0);
  }

  std::vector<std::pair<cv::Point2f, double>> first_removed_kpts_indexes;
  cv::Point2f second_kp;

  for (unsigned int i = 0; i < pts_raw.size(); i++) {
    int uu = (int)pts_raw[i].x;
    int vv = (int)pts_raw[i].y;

    if (grid.at<char>(vv, uu) != 1)
      continue;

    double second_score = -1;

    for (int k = -dist_thresh; k < (dist_thresh + 1); k++)
      for (int j = -dist_thresh; j < (dist_thresh + 1); j++) {
        if (j == 0 && k == 0)
          continue;
        int new_x = uu + j, new_y = vv + k;
        if (new_x < border || new_y < border || new_x > img_width - border ||
            new_y > img_height - border)
          continue;

        if (confidence.at<float>(vv + k, uu + j) <
            confidence.at<float>(vv, uu)) {
          grid.at<char>(vv + k, uu + j) = 0;
          if (second_score < confidence.at<float>(vv + k, uu + j)) {
            second_score = confidence.at<float>(vv + k, uu + j);
            second_kp = cv::Point2f(uu + j, vv + k);
          }
        }
      }
    if (second_score > 0)
      first_removed_kpts_indexes.push_back(
          std::make_pair(second_kp, second_score));
    grid.at<char>(vv, uu) = 2;
  }

  size_t valid_cnt = 0;

  for (int v = 0; v < (img_height); v++) {
    for (int u = 0; u < (img_width); u++) {
      if (u >= (img_width - border) || u < border ||
          v >= (img_height - border) || v < border)
        continue;

      if (grid.at<char>(v, u) == 2) {
        int select_ind = (int)inds.at<unsigned short>(v, u);
        float _conf = confidence.at<float>(v, u);
        cv::Point2f p = pts_raw[select_ind];
        pts_conf_vec.push_back(std::make_pair(p, _conf));
        valid_cnt++;
      }
    }
  }

  pts.reserve(max_num);
  std::sort(pts_conf_vec.begin(), pts_conf_vec.end(), pt_conf_comp);
  for (unsigned int i = 0; i < max_num && i < pts_conf_vec.size(); i++) {
    pts.push_back(cv::KeyPoint(pts_conf_vec[i].first, 1));
  }

  /*std::sort(first_removed_kpts_indexes.begin(),
            first_removed_kpts_indexes.end(), pt_conf_comp);

  if (pts.size() < max_num) {
      int num = std::min(max_num - pts.size(),
  first_removed_kpts_indexes.size()); for (int i = 0; i < num; i++)
  pts.push_back(first_removed_kpts_indexes[i].first);
  }
  first_removed_kpts_indexes.clear();*/
  pts_conf_vec.clear();
  printf(" NMS keypoints_no_nms %ld keypoints %ld\n", det.size(), pts.size());
}