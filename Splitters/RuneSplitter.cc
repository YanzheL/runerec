///
// Created by LI YANZHE on 18-4-13.
//

#include "RuneSplitter.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include "../Utils.h"

using namespace std;
using namespace cv;
using cuda::GpuMat;
using namespace runerec;
//using namespace Eigen;

void
RuneSplitter::split(const cv::Mat &frame, std::vector<cv::Mat> &blocks, vector<RotatedRect> &roi) {
  cuda::GpuMat src(frame);
  vector<GpuMat> gRes;
  split(src, gRes, roi);
  for (const auto &block:gRes) {
    Mat m;
    block.download(m);
    blocks.push_back(m);
  }
//    for (int i = 0; i < blocks.size(); ++i) {
//        auto block = blocks[i];
//        imshow(to_string(i), block);
//    }
}

void
RuneSplitter::getContours(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &erode_binary, cv::cuda::GpuMat &dilate_binary,
                          std::vector<std::vector<cv::Point2i>> &contours) {
  src.copyTo(erode_binary);
  if (erode_binary.channels() != 1) {
    cuda::cvtColor(erode_binary, erode_binary, COLOR_BGR2GRAY);//单通道
  }
//    cuda::threshold(erode_binary, erode_binary, 127, 255, THRESH_BINARY);
  cpuThreshold(erode_binary, erode_binary, 127, 255, THRESH_OTSU);
  Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
  Ptr<cuda::Filter> erode = cuda::createMorphologyFilter(cv::MORPH_ERODE, erode_binary.type(), element);
  erode->apply(erode_binary, erode_binary);
  Mat element2 = getStructuringElement(MORPH_RECT, Size(4, 4));
  Ptr<cuda::Filter> dilate = cuda::createMorphologyFilter(cv::MORPH_DILATE, erode_binary.type(), element2);
  dilate->apply(erode_binary, erode_binary);
  dilate->apply(erode_binary, dilate_binary);
//    dilate->apply(dilate_binary, dilate_binary);
//    dilate_binary=erode_binary;
  vector<Vec4i> hierarchy;
  Mat dld_dst;
  dilate_binary.download(dld_dst);
  findContours(dld_dst, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
}

void RuneSplitter::split(const cv::cuda::GpuMat &frame, std::vector<cv::cuda::GpuMat> &roi,
                         std::vector<cv::RotatedRect> &roi_rects) {

  GpuMat dst, erode_binary, dilate_binary;
  frame.copyTo(dst);
  vector<vector<Point2i>> contours;
  getContours(dst, erode_binary, dilate_binary, contours);
  split(frame, erode_binary, contours, roi, roi_rects);
}

//bool FireRuneSplitter::checkSudoku(const vector<vector<Point2i>> &contours,
//                                   vector<RotatedRect> &sudoku_rects) {
//    float ratio = width / height;
//    int sudoku = 0;
//    float low_threshold = 0.6;
//    float high_threshold = 1.4;
//    size_t i = 0;
//    RotatedRect rect;
//    vector<Point2f> centers;
//    sudoku_rects.clear();
//    for (; i < contours.size(); i++) {
//        rect = minAreaRect(contours[i]);
//        rect = adjustRRect(rect);
//        const Size2f &s = rect.size;
//        float ratio_cur = s.width / s.height;
//        if (ratio_cur > 0.8 * ratio && ratio_cur < 1.2 * ratio &&
//            s.width > low_threshold * width && s.width < high_threshold * width &&
//            s.height > low_threshold * height && s.height < high_threshold * height &&
//            ((rect.angle > -10 && rect.angle < 10) || rect.angle < -170 || rect.angle > 170)) {
//            sudoku_rects.push_back(rect);
//            centers.push_back(rect.center);
//            ++sudoku;
//        }
//    }
//    if (sudoku > 9) {
//        float dist_map[15][15] = {0};
//        // calculate distance of each cell center
//        for (int i = 0; i < sudoku; ++i) {
//            for (int j = i + 1; j < sudoku; ++j) {
//                float d = sqrt((centers[i].x - centers[j].x) * (centers[i].x - centers[j].x) +
//                               (centers[i].y - centers[j].y) * (centers[i].y - centers[j].y));
//                dist_map[i][j] = d;
//                dist_map[j][i] = d;
//            }
//        }
//        // choose the minimun distance cell as center cell
//        int center_idx = 0;
//        float min_dist = 100000000;
//        for (int i = 0; i < sudoku; ++i) {
//            float cur_d = 0;
//            for (int j = 0; j < sudoku; ++j) {
//                cur_d += dist_map[i][j];
//            }
//            if (cur_d < min_dist) {
//                min_dist = cur_d;
//                center_idx = i;
//            }
//        }
//        // sort distance between each cell and the center cell
//        vector<pair<float, int> > dist_center;
//        for (int i = 0; i < sudoku; ++i) {
//            dist_center.push_back(make_pair(dist_map[center_idx][i], i));
//        }
//        sort(dist_center.begin(), dist_center.end(),
//             [](const pair<float, int> &p1, const pair<float, int> &p2) { return p1.first < p2.first; });
//        // choose the nearest 9 cell as suduku
//        vector<RotatedRect> sudoku_rects_temp;
//        for (int i = 0; i < 9; ++i) {
//            sudoku_rects_temp.push_back(sudoku_rects[dist_center[i].second]);
//        }
//        sudoku_rects_temp.swap(sudoku_rects);
//    }
//    //cout << "sudoku n: " << sudoku_rects.size() << endl;
//    return !(sudoku_rects.size() < 9);
//}



