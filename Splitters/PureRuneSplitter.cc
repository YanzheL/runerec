//
// Created by LI YANZHE on 18-4-13.
//

#include "PureRuneSplitter.h"

using namespace std;
using namespace cv;
using cuda::GpuMat;

void PureRuneSplitter::split(const cv::cuda::GpuMat &frame, const cv::cuda::GpuMat &dst,
                             const std::vector<std::vector<cv::Point2i>> &contours,
                             std::vector<cv::cuda::GpuMat> &roi, std::vector<cv::RotatedRect> &roi_rects) {

    GpuMat src(frame);
//    if (frame.channels() != 1) {
//        cuda::cvtColor(src, src, CV_BGR2GRAY);//单通道
//    }
//    cuda::threshold(src, dst, 150, 255, THRESH_BINARY);//二值化
//    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
//    Ptr<cuda::Filter> erode = cuda::createMorphologyFilter(cv::MORPH_ERODE, dst.type(), element);
//    erode->apply(dst, dst);
//    Mat element2 = getStructuringElement(MORPH_RECT, Size(4, 4));
//    Ptr<cuda::Filter> dilate = cuda::createMorphologyFilter(cv::MORPH_DILATE, dst.type(), element2);
//    dilate->apply(dst, dst);
//    vector<vector<Point2i>> contours;
//    vector<Vec4i> hierarchy;
//    Mat dld_dst;
//    dst.download(dld_dst);
//    findContours(dld_dst, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    checkSudoku(contours, roi_rects);
//    filterRoi(dst,roi_rects);
    vector<Point2fWithIdx> _centers;
    for (size_t i = 0; i < roi_rects.size(); i++) {
        const RotatedRect &rect = roi_rects[i];
        Point2f vecs[4];
        rect.points(vecs);
        _centers.push_back({rect.center, i});
    }
    if (_centers.size() != 9)
        return;
    sort(_centers.begin(), _centers.end(),
         [](const Point2fWithIdx &p1, const Point2fWithIdx &p2) {
             return p1.p.y < p2.p.y;
         });
    sort(_centers.begin() + 0, _centers.begin() + 3,
         [](const Point2fWithIdx &p1, const Point2fWithIdx &p2) {
             return p1.p.x < p2.p.x;
         });
    sort(_centers.begin() + 3, _centers.begin() + 6,
         [](const Point2fWithIdx &p1, const Point2fWithIdx &p2) {
             return p1.p.x < p2.p.x;
         });
    sort(_centers.begin() + 6, _centers.end(),
         [](const Point2fWithIdx &p1, const Point2fWithIdx &p2) {
             return p1.p.x < p2.p.x;
         });
    cout << "centers size = " << _centers.size() << endl;
    cout << "roi_rects size = " << roi_rects.size() << endl;

    GpuMat img(frame);
//    cuda::threshold(frame, img, 127, 255, THRESH_BINARY_INV);
    img.convertTo(img, img.type(), -1.0 / 255.0, 1);
    vector<RotatedRect> ordered_roi_rects;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            size_t idx = i * 3 + j;
            size_t sudoku_idx = _centers[idx].idx;
            auto rr = roi_rects[sudoku_idx];
//            Rect cell_roi=rr.boundingRect();
//            Rect scale_roi = Rect(cell_roi.x + 10, cell_roi.y + 10, cell_roi.width - 15, cell_roi.height - 15);
            RotatedRect rc(rr.center, Size(rr.size.width - 10, rr.size.height - 10), rr.angle);
            ordered_roi_rects.push_back(rc);
        }
    }
    roi_rects.clear();
    roi_rects.assign(ordered_roi_rects.begin(), ordered_roi_rects.end());
    for (const auto &r:roi_rects) {
        Rect cell_roi = r.boundingRect();
//        Rect scale_roi = Rect(cell_roi.x + 10, cell_roi.y + 10, cell_roi.width - 15, cell_roi.height - 15);
        roi.push_back(img(cell_roi));
    }

//    for (size_t i = 0; i < 3; i++) {
//        for (size_t j = 0; j < 3; j++) {
//            size_t idx = i * 3 + j;
//
////            if (idx >= _centers.size() - 1)
////                break;
//            size_t sudoku_idx = _centers[idx].idx;
////            if (sudoku_idx >= roi_rects.size() - 1)
////                break;
//            Rect cell_roi = roi_rects[sudoku_idx].boundingRect();
//            Rect scale_roi = Rect(cell_roi.x + 5, cell_roi.y + 5, cell_roi.width - 9, cell_roi.height - 9);
////            if (scale_roi.tl().x < 0 || (scale_roi.br().x - cell_roi.width) < 0 || scale_roi.br().y > img.rows
////                || scale_roi.br().x > img.cols || (scale_roi.tl().x + cell_roi.width) > img.cols ||
////                (scale_roi.br().y - cell_roi.height) < 0)
////                break;
//            roi.push_back(img(scale_roi));
//        }
//    }
}

bool PureRuneSplitter::checkSudoku(const vector<vector<Point2i>> &contours,
                                   vector<RotatedRect> &sudoku_rects) {
    float ratio = roi_width / roi_height;
    int sudoku = 0;
    float low_threshold = 0.6;
    float high_threshold = 1.4;
    size_t i = 0;
    RotatedRect rect;
    vector<Point2f> centers;
    sudoku_rects.clear();
    for (; i < contours.size(); i++) {
        rect = minAreaRect(contours[i]);
        rect = adjustRRect(rect);
        const Size2f &s = rect.size;
        float ratio_cur = s.width / s.height;
        if (ratio_cur > 0.8 * ratio && ratio_cur < 1.2 * ratio &&
            s.width > low_threshold * roi_width && s.width < high_threshold * roi_width &&
            s.height > low_threshold * roi_height && s.height < high_threshold * roi_height &&
            ((rect.angle > -10 && rect.angle < 10) || rect.angle < -170 || rect.angle > 170)) {
            sudoku_rects.push_back(rect);
            centers.push_back(rect.center);
            ++sudoku;
        }
    }
    if (sudoku > 9) {
        float dist_map[15][15] = {0};
        // calculate distance of each cell center
        for (int i = 0; i < sudoku; ++i) {
            for (int j = i + 1; j < sudoku; ++j) {
                float d = sqrt((centers[i].x - centers[j].x) * (centers[i].x - centers[j].x) +
                               (centers[i].y - centers[j].y) * (centers[i].y - centers[j].y));
                dist_map[i][j] = d;
                dist_map[j][i] = d;
            }
        }
        // choose the minimun distance cell as center cell
        int center_idx = 0;
        float min_dist = 100000000;
        for (int i = 0; i < sudoku; ++i) {
            float cur_d = 0;
            for (int j = 0; j < sudoku; ++j) {
                cur_d += dist_map[i][j];
            }
            if (cur_d < min_dist) {
                min_dist = cur_d;
                center_idx = i;
            }
        }
        // sort distance between each cell and the center cell
        vector<pair<float, int> > dist_center;
        for (int i = 0; i < sudoku; ++i) {
            dist_center.push_back(make_pair(dist_map[center_idx][i], i));
        }
        sort(dist_center.begin(), dist_center.end(),
             [](const pair<float, int> &p1, const pair<float, int> &p2) { return p1.first < p2.first; });
        // choose the nearest 9 cell as suduku
        vector<RotatedRect> sudoku_rects_temp;
        for (int i = 0; i < 9; ++i) {
            sudoku_rects_temp.push_back(sudoku_rects[dist_center[i].second]);
        }
        sudoku_rects_temp.swap(sudoku_rects);
    }
    //cout << "sudoku n: " << sudoku_rects.size() << endl;
    return sudoku_rects.size() >= 9;
}