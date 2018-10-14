//
// Created by LI YANZHE on 18-4-13.
//

#ifndef HANDWRITING_FLAMEWORDS_MODULE_2_LEDRUNESPLITTER_H
#define HANDWRITING_FLAMEWORDS_MODULE_2_LEDRUNESPLITTER_H

#include "RuneSplitter.h"

class LedRuneSplitter : public RuneSplitter {
public:
    LedRuneSplitter() = default;

    LedRuneSplitter(float rune_w, float rune_h) : RuneSplitter(rune_w, rune_h) {};

    virtual void split(const cv::cuda::GpuMat &frame, const cv::cuda::GpuMat &dst,
                       const std::vector<std::vector<cv::Point2i>> &contours,
                       std::vector<cv::cuda::GpuMat> &roi, std::vector<cv::RotatedRect> &roi_rects);

protected:

    virtual bool
    checkSudoku(const std::vector<std::vector<cv::Point2i>> &contours, std::vector<cv::RotatedRect> &sudoku_rects);

};


#endif //HANDWRITING_FLAMEWORDS_MODULE_2_LEDRUNESPLITTER_H
