//
// Created by LI YANZHE on 18-4-13.
//

#ifndef RUNERECQT_SINGLETONFACTORY_H
#define RUNERECQT_SINGLETONFACTORY_H

#include <map>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <functional>

struct Point2fWithIdx {
  cv::Point2f p;
  size_t idx;
//        Point2fWithIdx(const cv::Point2f _p, size_t _idx) : p(_p), idx(_idx) {}
};

constexpr inline unsigned int BKDRHash(std::string &src) {
  char *str = (char *) src.c_str();
  unsigned int seed = 131; // 31 131 1313 13131 131313 etc..
  unsigned int hash = 0;
  while (*str) {
    hash = hash * seed + (*str++);
  }
  return (hash & 0x7FFFFFFF);
};

inline void showGpuMat(const std::string info, const cv::cuda::GpuMat &img) {
  cv::Mat dld;
  img.download(dld);
  cv::imshow(info, dld);
}

inline void
cvtToCpuBatch(const std::vector<cv::cuda::GpuMat> &gpu_batch, std::vector<cv::Mat> &cpu_batch, float factor = 1.0) {
  for (const auto &m:gpu_batch) {
    cv::Mat c_m;
    m.download(c_m);
    cpu_batch.push_back(c_m * factor);
  }
}

inline bool checkRoi(const cv::cuda::GpuMat &img, const cv::Rect &rect) {
  return rect.x >= 0
      && rect.y >= 0
      && rect.x > 0
      && rect.y > 0
      && rect.x + rect.width < img.cols
      && rect.y + rect.height < img.rows;
}

template<typename T, class F>
inline bool filter(std::vector<T> &vec, F func) {
  bool ret = false;
  for (auto it = vec.begin(); it != vec.end();) {
    if (!func(*it)) {
      it = vec.erase(it);
      ret = true;
    } else
      ++it;
  }
  return ret;
};

inline void filterRoi(const cv::cuda::GpuMat &img, std::vector<cv::RotatedRect> &rects) {
  filter(rects, [img](cv::RotatedRect &rr) { return checkRoi(img, rr.boundingRect()); });
}

inline void cpuThreshold(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, double thresh, double maxval, int type) {
  cv::Mat tp;
  src.download(tp);
  cv::threshold(tp, tp, thresh, maxval, type);
  dst.upload(tp);
}

inline void drawRects(cv::Mat &frame, const std::vector<cv::RotatedRect> &rects) {
  for (size_t i = 0; i < rects.size(); i++) {//画矩形
    const cv::RotatedRect &rect = rects[i];
    cv::Point2f vecs[4];
    rect.points(vecs);
    for (int j = 0; j < 4; j++)
      line(frame, vecs[j], vecs[(j + 1) % 4], cv::Scalar(0, 255, 0), 3);
  }
}

constexpr inline double accuracy(const int res[], const std::vector<int> &answer) {
  double t = 0;
  double s = answer.size();
  for (int i = 0; i < s; ++i) {
    if (res[i] == answer[i])
      ++t;
  }
  return t / s;
}

#endif //RUNERECQT_SINGLETONFACTORY_H
