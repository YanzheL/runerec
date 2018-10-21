//
// Created by LI YANZHE on 18-4-13.
//

#ifndef HANDWRITING_FLAMEWORDS_MODULE_2_PURERUNESPLITTER_H
#define HANDWRITING_FLAMEWORDS_MODULE_2_PURERUNESPLITTER_H

#include "RuneSplitter.h"

class PureRuneSplitter : public RuneSplitter {
 public:
  PureRuneSplitter() = default;

  PureRuneSplitter(float rune_w, float rune_h) : RuneSplitter(rune_w, rune_h) {};

  void split(const cv::cuda::GpuMat &frame, const cv::cuda::GpuMat &dst,
             const std::vector<std::vector<cv::Point2i>> &contours,
             std::vector<cv::cuda::GpuMat> &roi, std::vector<cv::RotatedRect> &roi_rects) override;

 protected:
  bool
  checkSudoku(const std::vector<std::vector<cv::Point2i>> &contours,
              std::vector<cv::RotatedRect> &sudoku_rects) override;

};

#endif //HANDWRITING_FLAMEWORDS_MODULE_2_PURERUNESPLITTER_H
