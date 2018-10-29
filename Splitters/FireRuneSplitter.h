//
// Created by LI YANZHE on 18-4-13.
//

#ifndef RUNEREC_FIRERUNESPLITTER_H
#define RUNEREC_FIRERUNESPLITTER_H

#include "RuneSplitter.h"
#include <Eigen/Dense>
namespace runerec {
class FireRuneSplitter : public RuneSplitter {
 public:
  FireRuneSplitter() = default;

  FireRuneSplitter(float rune_w, float rune_h) : RuneSplitter(rune_w, rune_h) {};

  FireRuneSplitter(float rune_w, float rune_h, float ratio) : RuneSplitter(rune_w, rune_h), resize_ratio(ratio) {};

  void split(const cv::cuda::GpuMat &frame, std::vector<cv::cuda::GpuMat> &roi,
             std::vector<cv::RotatedRect> &roi_rects) override;

  void split(const cv::cuda::GpuMat &frame, const cv::cuda::GpuMat &dst,
             const std::vector<std::vector<cv::Point2i>> &contours,
             std::vector<cv::cuda::GpuMat> &roi, std::vector<cv::RotatedRect> &roi_rects) override {};

 protected:
  bool
  checkSudoku(const std::vector<std::vector<cv::Point2i>> &contours,
              std::vector<cv::RotatedRect> &sudoku_rects) override;

 private:
  template<typename T>
  static std::vector<int> AscendingSort_Indexes(const std::vector<T> &v);

  template<typename T>
  static std::vector<int> DescendingSort_Indexes(const std::vector<T> &v);

  static void ExtractNineImage(cv::cuda::GpuMat &image, cv::cuda::GpuMat &dst, int number);

  static void rectSplit(cv::Point2f &tl, cv::Point2f &tr, cv::Point2f &bl, cv::Point2f &br,
                        std::vector<cv::RotatedRect> &rects, int dim);

  static void splitLine(cv::Point2f &begin, cv::Point2f &end, int n_part, std::vector<cv::Point2f> &split_points);

  static cv::RotatedRect resizeRect(cv::RotatedRect &rect, float factor_x, float factor_y);

  static bool filterValidSideBoxes(std::vector<cv::RotatedRect> &rects);

  static void shrinkImg(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, float ratio_x, float ratio_y);

 private:

  bool ChooseFourPoints();

  bool Filter();

  void Spilt(std::vector<cv::RotatedRect> &sudoku_rects);

  void PerspectiveTransform(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);

  void InversePerspectiveTransform();

  void Crop(cv::cuda::GpuMat &image, cv::cuda::GpuMat &dst, float thresholds);

//    void PostProcessing(cv::cuda::GpuMat &image, cv::cuda::GpuMat &dst, double parameter_1,
//                        double parameter_2);


 private:
  cv::RNG rng_sec;
  std::function<bool(float, float)> AscendingSort = [](float i, float j) { return (i < j); };
  std::function<bool(float, float)> DescendingSort = [](float i, float j) { return (i > j); };
 private:
  float resize_ratio = 1;
  cv::Mat perspective_matrix, inverse_per_matrix;
  Eigen::Matrix3f per_matrix, inverse_perspective_matrix;
  std::vector<cv::RotatedRect> LeftFiveRec;
  std::vector<cv::RotatedRect> RightFiveRec;
  std::vector<int> Index;
  float height;
  cv::RotatedRect TopLeftRec, TopRightRec, BottomLeftRec, BottomRightRec;
  std::vector<float> TopLeftCorner_x;
  std::vector<float> TopLeftCorner_y;
  cv::Point2f TopLeft, TopRight, BottomLeft, BottomRight;
  cv::Point2f Point_1, Point_2, Point_3, Point_4;
  cv::Point2f Top_Middle_First, Top_Middle_Sec, Left_Middle_First, Left_Middle_Sec, Bottom_Middle_First,
      Bottom_Middle_Sec, Right_Middle_First, Right_Middle_Sec;
};
}

#endif //RUNEREC_FIRERUNESPLITTER_H
