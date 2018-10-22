//
// Created by LI YANZHE on 18-4-13.
//

#ifndef RUNEREC_LEDRUNESPLITTER_H
#define RUNEREC_LEDRUNESPLITTER_H

#include "RuneSplitter.h"

namespace runerec {
class LedRuneSplitter : public RuneSplitter {
 public:
  LedRuneSplitter() = default;

  LedRuneSplitter(float rune_w, float rune_h) : RuneSplitter(rune_w, rune_h) {};

  void split(const cv::cuda::GpuMat &frame, const cv::cuda::GpuMat &dst,
             const std::vector<std::vector<cv::Point2i>> &contours,
             std::vector<cv::cuda::GpuMat> &roi, std::vector<cv::RotatedRect> &roi_rects) override;

 protected:

  bool
  checkSudoku(const std::vector<std::vector<cv::Point2i>> &contours,
              std::vector<cv::RotatedRect> &sudoku_rects) override;

};
}

#endif //RUNEREC_LEDRUNESPLITTER_H
