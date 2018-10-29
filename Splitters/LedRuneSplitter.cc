//
// Created by LI YANZHE on 18-4-13.
//


#include "LedRuneSplitter.h"
#include "../Utils.h"
//#include <opencv2/opencv.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#include <opencv2/cudafilters.hpp>
//#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
//#include <opencv2/cudafeatures2d.hpp>
using namespace std;
using namespace cv;
using cuda::GpuMat;
using namespace runerec;

void LedRuneSplitter::split(const cv::cuda::GpuMat &frame, const cv::cuda::GpuMat &dst,
                            const std::vector<std::vector<cv::Point2i>> &contours,
                            std::vector<cv::cuda::GpuMat> &roi, std::vector<cv::RotatedRect> &roi_rects) {
//    Mat dld_src;
//    frame.download(dld_src);
//    Mat element2 = getStructuringElement(MORPH_RECT, Size(4, 4));
//    Ptr<cuda::Filter> dilate = cuda::createMorphologyFilter(cv::MORPH_DILATE, dst.type(), element2);
//    dilate->apply(dst, dst);
  showGpuMat("led dst", dst);
//    Mat show;
//    frame.download(show);
//    for(int i=0;i<contours.size();++i){
//        drawContours(show,contours,i,CV_RGB(0,255,255),4);
//    }
//    imshow("contours",show);
  checkSudoku(contours, roi_rects);
//    Mat show;
//    frame.download(show);
//    drawRects(show,roi_rects);
//    imshow("led crop",show);
//    waitKey(10);
  filterRoi(dst, roi_rects);
//    filter(roi_rects,[frame](RotatedRect &rect){return rect.boundingRect().y<=frame.rows/3.0;});
//    if (roi_rects.size()==5 ) {
//        for (size_t i = 0; i < roi_rects.size(); i++) {
//            const RotatedRect &rect = roi_rects[i];
//            Point2f vecs[4];
//            rect.points(vecs);
//            circle(dld_src, vecs[3], 5, Scalar(255, 255, 255), -1);
//            for (int i = 0; i < 4; i++)
//                line(dld_src, vecs[i], vecs[(i + 1) % 4], Scalar(255, 255, 255), 3);
//        }
//        imshow("dld_src", dld_src);
//        waitKey(0);
  for (const auto &r:roi_rects) {
    GpuMat cell = dst(r.boundingRect());
    cuda::threshold(cell, cell, 127, 1, THRESH_BINARY);
    roi.push_back(cell);
  }
//    }
}

bool LedRuneSplitter::checkSudoku(const std::vector<std::vector<cv::Point2i>> &contours,
                                  std::vector<cv::RotatedRect> &sudoku_rects) {
//    float low_threshold = 0.6;
//    float high_threshold = 1.4;
  float low_threshold = 0.2;
  float high_threshold = 2.4;
  for (size_t i = 0; i < contours.size(); i++) {
    RotatedRect rect = minAreaRect(contours[i]);
    Rect box = rect.boundingRect();
    const Size2f &t = box.size();
    double cur_ratio = t.width / t.height;
    if (
//                cur_ratio>=02
        cur_ratio < 1
            //            t.width > low_threshold * roi_width
            && t.width < high_threshold * roi_width
                //            && t.height > low_threshold * roi_height
            && t.height < high_threshold * roi_height
            && box.y + box.height < 240
//            && rect.tl().y < 240 || t.width > 5 && t.width < 12 && t.height > low_threshold * roi_height &&
//                                    t.height < high_threshold * roi_height && rect.tl().y < 240
        ) {
//            ledrect = minAreaRect(contours[i]);
//            ledrect = adjustRRect(ledrect);
      sudoku_rects.push_back(rect);
    }
  }
  sort(sudoku_rects.begin(), sudoku_rects.end(),
       [](const RotatedRect &r1, const RotatedRect &r2) { return r1.boundingRect().x <= r2.boundingRect().x; });
  return sudoku_rects.size() == 5;
}
