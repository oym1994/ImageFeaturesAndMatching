#include <opencv2/opencv.hpp>
#include <vector>
class ContourEtractor : public cv::DescriptorExtractor
{
public:
    void detect(cv::InputArray _image, CV_OUT std::vector<cv::KeyPoint> &keypoints,
                cv::InputArray _mask = cv::noArray())
    {
        cv::Mat image = _image.getMat().clone(), mask = _mask.getMat(), original_image = _image.getMat();
        cv::threshold(image, image, 80, 255, cv::THRESH_BINARY);
        cv::Mat canny_image;
        cv::imshow("binary map", image);
        cv::Canny(image, canny_image, 200, 500, 3, false);
        cv::imshow("canny_image", canny_image);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(canny_image, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
        for (const auto &contour : contours)
        {
            if (cv::contourArea(contour) < 50)
                continue;
            // if(contourArea(contours[index]))
            std::cout << "contour point size: " << contour.size() << "   "
                      << ",  area: " << cv::contourArea(contour) << std::endl;
            for (const auto point : contour)
                keypoints.push_back(cv::KeyPoint(point, 1));
        }
        std::cout << "extracted " << keypoints.size() << " contour points" << std::endl;

        cv::Mat polyPic = cv::Mat::zeros(original_image.size(), CV_8UC3);

        for (const auto &contour : contours)
        {
            std::vector<std::vector<cv::Point>> onecontour{contour};
            if (cv::contourArea(contour) < 50)
                cv::drawContours(polyPic, onecontour, -1, cv::Scalar(255, 255, 255), 1);
            else
                cv::drawContours(polyPic, onecontour, -1, cv::Scalar(rand() & 255, rand() & 255, rand() & 255), 2);
        }

        // cv::drawContours(original_image, contours, -1, cv::Scalar(100, 200, 250));
        cv::imshow("contours", polyPic);
        cv::waitKey(0);
    }

    CV_WRAP static cv::Ptr<cv::DescriptorExtractor> create()
    {
        return new ContourEtractor();
    }
};