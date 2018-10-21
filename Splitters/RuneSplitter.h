//
// Created by LI YANZHE on 18-4-13.
//

#ifndef RUNE_RUNESPLITER_H
#define RUNE_RUNESPLITER_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <utility>
#include <string>

#ifdef USE_EIGEN
#include <Eigen/Core>
#include <Eigen/Dense>
#endif

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>
#include "../Utils.h"
#include <sstream>

#define WIDTH 1140
#define HEIGHT 710

namespace runerec {

class RuneSplitter {
 public:

  RuneSplitter() : roi_width(90), roi_height(50) {}

  RuneSplitter(float rune_w, float rune_h) : roi_width(rune_w), roi_height(rune_h) {}

  static void
  getContours(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &erode_binary, cv::cuda::GpuMat &dilate_binary,
              std::vector<std::vector<cv::Point2i>> &contours);

  virtual void split(const cv::Mat &frame, std::vector<cv::Mat> &blocks, std::vector<cv::RotatedRect> &roi);

  virtual void split(const cv::cuda::GpuMat &frame, std::vector<cv::cuda::GpuMat> &roi,
                     std::vector<cv::RotatedRect> &roi_rects);

  virtual void split(const cv::cuda::GpuMat &frame, const cv::cuda::GpuMat &dst,
                     const std::vector<std::vector<cv::Point2i>> &contours,
                     std::vector<cv::cuda::GpuMat> &roi, std::vector<cv::RotatedRect> &roi_rects) = 0;

 protected:

  inline static cv::RotatedRect adjustRRect(const cv::RotatedRect &rect) {
    const cv::Size2f &s = rect.size;
    if (s.width > s.height)
      return rect;
    return cv::RotatedRect(rect.center, cv::Size2f(s.height, s.width), rect.angle + 90.0);
  }

  virtual bool
  checkSudoku(const std::vector<std::vector<cv::Point2i>> &contours, std::vector<cv::RotatedRect> &sudoku_rects) = 0;

 protected:
  float roi_width;
  float roi_height;
};
}

#endif //RUNE_RUNESPLITER_H
