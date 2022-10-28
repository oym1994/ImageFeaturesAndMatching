#include "keypoints_detection.hpp"

using namespace cv;

void processDescriptor(cv::Mat &desc,
                       const std::vector<cv::KeyPoint> &keypoints,
                       const int width, const int height)
{

  float center_x = width * 0.5, center_y = height * 0.5;

  float scale = 1.0 / (width * 0.7);

  cv::Mat kpts = cv::Mat(keypoints.size(), 2, CV_32F),
          scores = cv::Mat(keypoints.size(), 1, CV_32F);
  for (int i = 0; i < keypoints.size(); i++)
  {
    kpts.at<float>(i, 0) = (keypoints[i].pt.x - center_x) * scale;
    kpts.at<float>(i, 1) = (keypoints[i].pt.y - center_y) * scale;
    scores.at<float>(i, 0) = keypoints[i].response;
  }

  cv::hconcat(std::vector<cv::Mat>{desc, kpts, scores}, desc);
}

int main(int argc, const char *argv[])
{

#ifdef USE_DEEP_FEATURES
  torch::autograd::GradMode::set_enabled(false);
#endif

  std::string keypointsType = "SuperPoint", // BRISK, ORB, AKAZE, SIFT, SURF, CONTOUR
      descriptorType = "SuperPoint",        // BRIEF, ORB, FREAK, AKAZE, SIFT, BRISK, SURF, SuperPoint
      matchType = "BF_L2";                  // BF_HAMMING, BF_L2, SuperGlue

  if (argc >= 2)
    keypointsType = std::string(argv[1]);
  if (argc >= 3)
    descriptorType = std::string(argv[2]);
  if (argc >= 4)
    matchType = std::string(argv[3]);

  std::cout << "keypointsType: " << keypointsType << std::endl;
  std::cout << "descriptorType: " << descriptorType << std::endl;
  std::cout << "matchType: " << matchType << std::endl;

  FeatureDetectorAndMatcher image_matching(keypointsType, descriptorType, matchType);

  cv::Mat image1 = cv::imread("../images/1.png", IMREAD_GRAYSCALE),
          image2 = cv::imread("../images/2.png", IMREAD_GRAYSCALE);

  cv::Mat desc1, desc2;
  std::vector<KeyPoint> keypoints1, keypoints2;
  image_matching.detectorAndCompute(image1, keypoints1, desc1);
  image_matching.detectorAndCompute(image2, keypoints2, desc2);

  // just for superglue match
  if (matchType == "SuperGlue")
  {
    processDescriptor(desc1, keypoints1, image1.cols, image1.rows);
    processDescriptor(desc2, keypoints2, image2.cols, image2.rows);
  }

  std::vector<cv::DMatch> matches;
  image_matching.matchFeatures(desc1, desc2, matches, "SEL_NN");

  // visualization
  cv::Mat image_matches;
  cv::drawMatches(image1, keypoints1, image2, keypoints2, matches,
                  image_matches);
  cv::putText(image_matches,
              "kpts_num: " + std::to_string(keypoints1.size()) + " / " +
                  std::to_string(keypoints1.size()),
              cv::Point(20, 30), 1, 1.0, cv::Scalar(0, 0, 255));
  cv::putText(image_matches, "match_num: " + std::to_string(matches.size()),
              cv::Point(20, 60), 1, 1.0, cv::Scalar(255, 0, 0));
  cv::imshow("Image Match", image_matches);
  cv::waitKey(0);

  return 0;
}
