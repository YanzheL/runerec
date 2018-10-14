//
// Created by LI YANZHE on 18-4-13.
//

#include "FireRuneSplitter.h"

using namespace std;
using namespace cv;
using cuda::GpuMat;

void FireRuneSplitter::split(const cv::cuda::GpuMat &frame, std::vector<cv::cuda::GpuMat> &roi,
                             vector<RotatedRect> &roi_rects) {
    roi_rects.clear();
    float hw_ratio = (float) frame.cols / frame.rows;
    resize_ratio = (float) frame.rows / 1080;
    GpuMat src(frame), dst, dilate_binary;
    cuda::resize(src, src, Size((int) 1080 * hw_ratio, 1080));
    vector<vector<Point2i>> contours;
//    getContours(src,dst,contours);
    if (frame.channels() != 1)
        cuda::cvtColor(src, src, CV_BGR2GRAY);
    {
//        cuda::threshold(src, dst, 127, 255, THRESH_BINARY);
        cpuThreshold(src, dst, 127, 255, THRESH_OTSU);
        Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
        Ptr<cuda::Filter> erode = cuda::createMorphologyFilter(MORPH_ERODE, dst.type(), element);
        erode->apply(dst, dst);
        Mat element2 = getStructuringElement(MORPH_RECT, Size(4, 4));
        Ptr<cuda::Filter> dilate = cuda::createMorphologyFilter(cv::MORPH_DILATE, dst.type(), element2);
        dilate->apply(dst, dilate_binary);
        vector<Vec4i> hierarchy;
        Mat dld_dst;
        dilate_binary.download(dld_dst);
        findContours(dld_dst, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//只检测最外测轮廓
    }
    checkSudoku(contours, roi_rects);
//    double parameter_1 = 70;
//    double parameter_2 = 0.05;
//    Mat drawing = Mat::zeros(dst.size(), CV_8UC3);
//    for (int i = 0; i < contours.size(); i++) {
//        Scalar color = Scalar(255, 255, 255);//rng.uniform 随机数生成
//        drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
//    }
//    imshow("ct",drawing);

//    Mat show1;
//    dst.download(show1);
//    cout << "sudoku_rects len = " << roi_rects.size() << endl;
//    drawRects(show1, roi_rects);
//    imshow("dst", show1);
//    waitKey(10);
    if (filterValidSideBoxes(roi_rects)) {
//        Mat show2;
//        dst.download(show2);
//        drawRects(show2, roi_rects);
//        imshow("filtered", show2);
//        waitKey(10);
        Spilt(roi_rects); //Separate ten rectangles into five left ones and right five ones.
        if (ChooseFourPoints()) {
            roi_rects.clear();
            rectSplit(TopLeft, TopRight, BottomLeft, BottomRight, roi_rects, 3);
            filterRoi(dst, roi_rects);
            if (roi_rects.size() != 9)
                return;
            GpuMat perspective, inverse_perspective;
//            showGpuMat("src_persp", src);
            PerspectiveTransform(src, perspective); // Implement perspective transformation.

//            perspective.convertTo(perspective, perspective.type(),3);
//            showGpuMat("perspective", perspective);
            InversePerspectiveTransform(); // inverse transformation
            vector<GpuMat> numbers;
            vector<GpuMat> cropped_numbers;
            for (int i = 0; i < 9; i++) {
                GpuMat number, cropped_number;
//                GpuMat processed_number;
                ExtractNineImage(perspective, number, i); // Separate the whole image into nine subimages.
                numbers.push_back(number);
                shrinkImg(number, cropped_number, 0.4, 0.2);
//                Crop(number, cropped_number, 3500); // Crop the digits in the image.
//                PostProcessing(cropped_number, processed_number, parameter_1, parameter_2);
//                cropped_number.convertTo(cropped_number, cropped_number.type(), 1.0 / 255.0);
//                showGpuMat("cropped_number", cropped_number);
                cropped_numbers.push_back(cropped_number);
//                cuda::threshold(processed_number, processed_number, 127, 1, THRESH_BINARY);
                roi.push_back(cropped_number);
            }
        }
    }
    for (int i = 0; i < roi_rects.size(); ++i)
        roi_rects[i] = resizeRect(roi_rects[i], resize_ratio, resize_ratio);

}

bool FireRuneSplitter::ChooseFourPoints() {
    TopLeftCorner_x.clear();
    for (size_t i = 0; i < LeftFiveRec.size(); i++) {//画矩形
        const RotatedRect &rect = LeftFiveRec[i];
        Point2f vecs[4];
        rect.points(vecs);
        TopLeftCorner_x.push_back(vecs[0].x);
        TopLeftCorner_y.push_back(vecs[1].y);
    }
    Index.clear();
    Index = AscendingSort_Indexes(TopLeftCorner_y);
    sort(TopLeftCorner_x.begin(), TopLeftCorner_x.end(), AscendingSort);
    sort(TopLeftCorner_y.begin(), TopLeftCorner_y.end(), DescendingSort);
    if (Filter())
        return false;
    TopLeftRec = LeftFiveRec[Index[0]];
    BottomLeftRec = LeftFiveRec[Index[4]];
    const RotatedRect &topleftrec = TopLeftRec;
    Point2f topleftrec_points[4];
    topleftrec.points(topleftrec_points);
    const RotatedRect &bottomleftrec = BottomLeftRec;
    Point2f bottomleftrec_points[4];
    bottomleftrec.points(bottomleftrec_points);
    TopLeftCorner_x.clear();
    TopLeftCorner_y.clear();
    for (size_t i = 0; i < RightFiveRec.size(); i++) {//画矩形
        const RotatedRect &rect = RightFiveRec[i];
        Point2f vecs[4];
        rect.points(vecs);
        TopLeftCorner_x.push_back(vecs[0].x);
        TopLeftCorner_y.push_back(vecs[1].y);
    }
    Index.clear();
    Index = AscendingSort_Indexes(TopLeftCorner_y);
    sort(TopLeftCorner_x.begin(), TopLeftCorner_x.end(), AscendingSort);
    sort(TopLeftCorner_y.begin(), TopLeftCorner_y.end(), DescendingSort);
    if (Filter())
        return false;
    TopRightRec = RightFiveRec[Index[0]];
    BottomRightRec = RightFiveRec[Index[4]];
    const RotatedRect &toprightrec = TopRightRec;
    Point2f toprightrec_points[4];
    toprightrec.points(toprightrec_points);
    const RotatedRect &bottomrightrec = BottomRightRec;
    Point2f bottomrightrec_points[4];
    bottomrightrec.points(bottomrightrec_points);
    height = abs(bottomrightrec_points[1].y - bottomrightrec_points[0].y);
    Point_1 = Point2f(topleftrec_points[2].x, topleftrec_points[2].y);
    Point_2 = Point2f(toprightrec_points[1].x, toprightrec_points[1].y);
    Point_3 = Point2f(bottomleftrec_points[3].x, bottomleftrec_points[3].y);
    Point_4 = Point2f(bottomrightrec_points[0].x, bottomrightrec_points[0].y);
    TopLeft = Point2f(topleftrec_points[2].x, topleftrec_points[2].y - height);
    TopRight = Point2f(toprightrec_points[1].x, toprightrec_points[1].y - height);
    BottomLeft = Point2f(bottomleftrec_points[3].x, bottomleftrec_points[3].y + height);
    BottomRight = Point2f(bottomrightrec_points[0].x, bottomrightrec_points[0].y + height);
    Top_Middle_First = Point2f(TopLeft.x + 1 / 3.0 * abs(TopLeft.x - TopRight.x),
                               TopLeft.y - 1 / 3.0 * abs(TopLeft.y - TopRight.y));
    Top_Middle_Sec = Point2f(TopLeft.x + 2 / 3.0 * abs(TopLeft.x - TopRight.x),
                             TopLeft.y - 2 / 3.0 * abs(TopLeft.y - TopRight.y));
    Left_Middle_First = Point2f(BottomLeft.x + 1 / 3.0 * abs(BottomLeft.x - TopLeft.x),
                                BottomLeft.y - 1 / 3.0 * abs(BottomLeft.y - TopLeft.y));
    Left_Middle_Sec = Point2f(BottomLeft.x + 2 / 3.0 * abs(BottomLeft.x - TopLeft.x),
                              BottomLeft.y - 2 / 3.0 * abs(BottomLeft.y - TopLeft.y));

    Bottom_Middle_First = Point2f(BottomLeft.x + 1 / 3.0 * abs(BottomLeft.x - BottomRight.x),
                                  BottomLeft.y - 1 / 3.0 * abs(BottomLeft.y - BottomRight.y));
    Bottom_Middle_Sec = Point2f(BottomLeft.x + 2 / 3.0 * abs(BottomLeft.x - BottomRight.x),
                                BottomLeft.y - 2 / 3.0 * abs(BottomLeft.y - BottomRight.y));

    Right_Middle_First = Point2f(BottomRight.x - 1 / 3.0 * abs(BottomRight.x - TopRight.x),
                                 BottomRight.y - 1 / 3.0 * abs(BottomRight.y - TopRight.y));
    Right_Middle_Sec = Point2f(BottomRight.x - 2 / 3.0 * abs(BottomRight.x - TopRight.x),
                               BottomRight.y - 2 / 3.0 * abs(BottomRight.y - TopRight.y));
    return true;
}

bool FireRuneSplitter::Filter() {
    return abs(TopLeftCorner_x[0] - TopLeftCorner_x[4]) > 50.0;
}

void FireRuneSplitter::PerspectiveTransform(cv::cuda::GpuMat &src, GpuMat &dst) {
    Point2f pts1[] = {TopLeft, BottomLeft, BottomRight, TopRight};
    Point2f pts2[] = {Point2f(0, 0), Point2f(0, HEIGHT), Point2f(WIDTH, HEIGHT), Point2f(WIDTH, 0)};
//    Point2f pts2[] = {Point2f(0, 0), Point2f(0, final.rows), Point2f(final.cols, final.rows), Point2f(final.cols, 0)};
    perspective_matrix = getPerspectiveTransform(pts1, pts2);
    inverse_per_matrix = getPerspectiveTransform(pts2, pts1);
    cuda::warpPerspective(src, dst, perspective_matrix, src.size(), INTER_LINEAR);
}

void FireRuneSplitter::InversePerspectiveTransform() {
    inverse_perspective_matrix << inverse_per_matrix.at<double>(0, 0),
            inverse_per_matrix.at<double>(0, 1),
            inverse_per_matrix.at<double>(0, 2),
            inverse_per_matrix.at<double>(1, 0),
            inverse_per_matrix.at<double>(1, 1),
            inverse_per_matrix.at<double>(1, 2),
            inverse_per_matrix.at<double>(2, 0),
            inverse_per_matrix.at<double>(2, 1),
            inverse_per_matrix.at<double>(2, 2);
//    cout << "Inverse_perspective_matrix\n" << inverse_perspective_matrix << endl;
    vector<Point2f> points;
    for (int i = 0; i < 16; i++) {
        Eigen::Vector3f point_result, point_origin;
        point_result << (WIDTH / 3.0) * (i % 4), (HEIGHT / 3.0) * (i / 4), 1;
        point_origin = inverse_perspective_matrix * point_result;
        Point2f result = Point2f(point_origin(0) / point_origin(2), point_origin(1) / point_origin(2));
        points.push_back(result);
    }

//    for(int i=0;i<9;++i){
//        int tl=i;
//        int tr=tl+1;
//        int bl=i+4;
//        int br=bl+1;
//        vector<Point2f> quad={points[tl],points[tr],points[bl],points[br]};
//        digit_rects.push_back(minAreaRect(quad));
//    }

}

void FireRuneSplitter::ExtractNineImage(cv::cuda::GpuMat &image, cv::cuda::GpuMat &dst, int number) {
    Rect Rectangle;
    Rectangle.x = (number % 3) * (WIDTH / 3);
    Rectangle.y = (number / 3) * (HEIGHT / 3);
    Rectangle.width = WIDTH / 3;
    Rectangle.height = HEIGHT / 3;
    dst = image(Rectangle);
}


void FireRuneSplitter::Crop(cv::cuda::GpuMat &image, cv::cuda::GpuMat &dst, float thresholds) {
    cuda::GpuMat threshold_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    cuda::threshold(image, threshold_output, 50, 255, THRESH_BINARY);
    Mat dld_src;
    threshold_output.download(dld_src);
    findContours(dld_src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    double min_contour_area = thresholds;
    for (auto it = contours.begin(); it != contours.end();) {
        if (contourArea(*it) < min_contour_area)
            it = contours.erase(it);
        else
            ++it;
    }
    if (contours.size() > 0) {
        vector<vector<Point> > contours_poly(contours.size());
        vector<Rect> boundRect(contours.size());
        for (size_t i = 0; i < contours.size(); i++) {
            approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
            boundRect[i] = boundingRect(Mat(contours_poly[i]));
        }
        Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
        for (size_t i = 0; i < contours.size(); i++) {
            Scalar color = Scalar(rng_sec.uniform(0, 255), rng_sec.uniform(0, 255), rng_sec.uniform(0, 255));
            drawContours(drawing, contours_poly, (int) i, color, 1, 8, vector<Vec4i>(), 0, Point());
            rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
        }
        if (contours.size() == 1) {
            Rect Number;
            Number.x = boundRect[0].tl().x;
            Number.y = boundRect[0].tl().y;
            Number.width = boundRect[0].br().x - boundRect[0].tl().x;
            Number.height = boundRect[0].br().y - boundRect[0].tl().y;
            dst = image(Number);
            return;
        } else {
            cout << "dst = drawing" << endl;
            dst = GpuMat(drawing);
            return;
        }

    }
    dst = image;
}

void FireRuneSplitter::Spilt(vector<RotatedRect> &sudoku_rects) {
    for (size_t i = 0; i < sudoku_rects.size(); i++) {//画矩形
        const RotatedRect &rect = sudoku_rects[i];
        Point2f vecs[4];
        rect.points(vecs);
        TopLeftCorner_x.push_back(vecs[0].x);
    }
    // Sort rectangles along with their indexes
    Index = AscendingSort_Indexes(TopLeftCorner_x);
    sort(TopLeftCorner_x.begin(), TopLeftCorner_x.end(), AscendingSort);
    for (size_t i = 0; i < sudoku_rects.size(); i++) {
        if (i < 5)
            LeftFiveRec.push_back(sudoku_rects[Index[i]]);
        else
            RightFiveRec.push_back(sudoku_rects[Index[i]]);
    }
}

template<typename T>
std::vector<int> FireRuneSplitter::AscendingSort_Indexes(const std::vector<T> &v) {

    // initialize original index locations
    vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(),
              [&v](int i1, int i2) { return v[i1] < v[i2]; });

    return idx;
}

template<typename T>
vector<int> FireRuneSplitter::DescendingSort_Indexes(const vector<T> &v) {

    // initialize original index locations
    vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(),
              [&v](int i1, int i2) { return v[i1] > v[i2]; });

    return idx;
}

bool FireRuneSplitter::checkSudoku(const vector<vector<Point2i>> &contours,
                                   vector<RotatedRect> &sudoku_rects) {

    cout << "----------------------------contours size = " << contours.size() << endl;
    if (contours.size() < 10)
        return false;
    float ratio = 140.0 / 69.4;
    int sudoku = 0;
    float low_threshold = 0.6;
    float high_threshold = 1.4;
//    float low_threshold = 0.1;
//    float high_threshold = 200000;
    size_t i = 0;
    RotatedRect rect;
    for (; i < contours.size(); i++) {
        rect = minAreaRect(contours[i]);
        rect = adjustRRect(rect);
        const Size2f &s = rect.size;
//        cout << "width = " << s.width << "    height = " << s.height << endl;
        float ratio_cur = s.width / s.height;
//        cout << "ratio cur = " << ratio_cur << endl;
        if (ratio_cur > 0.4 * ratio
            && ratio_cur < 1.6 * ratio
            && s.width > low_threshold * roi_width
            && s.width < high_threshold * roi_width
            && s.height > low_threshold * roi_height
            && s.height < high_threshold * roi_height
            && ((rect.angle > -10 && rect.angle < 10) || rect.angle < -170 || rect.angle > 170)
//               && ((rect.angle > -20 && rect.angle < 20) || rect.angle < -170 || rect.angle > 170)
                ) {
//            cout << "Bingo" << endl;
            sudoku_rects.push_back(rect);
            ++sudoku;
        }

    }
    cout << "sudoku_rects =  " << sudoku << "  i" << i << endl;
    return sudoku >= 9;
}

void FireRuneSplitter::rectSplit(cv::Point2f &tl, cv::Point2f &tr, cv::Point2f &bl, cv::Point2f &br,
                                 std::vector<cv::RotatedRect> &rects, int dim) {
    vector<Point2f> tls, trs, bls, brs;
    vector<Point2f> left_split_pts, right_split_pts;
    splitLine(tl, bl, dim, left_split_pts);
    splitLine(tr, br, dim, right_split_pts);
    //horizon split
    for (int i = 0; i <= dim; ++i) {
        vector<Point2f> splits;
        splitLine(left_split_pts[i], right_split_pts[i], dim, splits);
        const size_t split_size = splits.size();
        for (int j = 0; j < split_size; ++j) {
            Point2f p = splits[j];
            if (i != 0) {
                if (j != 0)
                    brs.push_back(p);
                if (j != split_size - 1)
                    bls.push_back(p);
            }
            if (i != dim && j != split_size - 1)
                tls.push_back(p);

            if (i != dim && j != 0)
                trs.push_back(p);
        }
    }
//    cout<<"tls size = "<<tls.size()<<endl;
//    cout<<"trs size = "<<trs.size()<<endl;
//    cout<<"bls size = "<<bls.size()<<endl;
//    cout<<"brs size = "<<brs.size()<<endl;
    for (int i = 0; i < dim * dim; ++i) {
        vector<Point2f> quadrilateral = {tls[i], trs[i], brs[i], bls[i]};
        rects.push_back(minAreaRect(quadrilateral));
    }
}

void
FireRuneSplitter::splitLine(cv::Point2f &begin, cv::Point2f &end, int n_part, std::vector<cv::Point2f> &split_points) {
    float step_x = (end.x - begin.x) / n_part;
    float step_y = (end.y - begin.y) / n_part;
    for (int i = 0; i < n_part; ++i)
        split_points.push_back({begin.x + i * step_x, begin.y + i * step_y});
    split_points.push_back(end);
}

RotatedRect FireRuneSplitter::resizeRect(cv::RotatedRect &rect, float factor_x, float factor_y) {
    Point2f pts[4];
    rect.points(pts);
    vector<Point2f> n_pts;
    for (int i = 0; i < 4; ++i) {
        Point2f tp;
        tp.x = pts[i].x * factor_x;
        tp.y = pts[i].y * factor_y;
        n_pts.push_back(tp);
    }
    return minAreaRect(n_pts);
}

bool FireRuneSplitter::filterValidSideBoxes(std::vector<cv::RotatedRect> &rects) {
    if (rects.size() < 10)
        return false;
    sort(rects.begin(), rects.end(),
         [](const RotatedRect &r1, const RotatedRect &r2) { return r1.boundingRect().x <= r2.boundingRect().x; });
    if (rects.size() == 10)
        return true;
    else {
        vector<pair<float, int>> dist;
        for (int i = 0; i < rects.size() - 1; ++i) {
            dist.push_back(make_pair(abs(rects[i + 1].boundingRect().x - rects[i].boundingRect().x), i));
        }
        sort(dist.begin(), dist.end(),
             [](const pair<float, int> &o1, const pair<float, int> &o2) { return o1.first <= o2.first; });
        vector<pair<float, int>> candidates;
        candidates.assign(dist.begin(), dist.begin() + 9);
        sort(candidates.begin(), candidates.end(),
             [](const pair<float, int> &o1, const pair<float, int> &o2) { return o1.second <= o2.second; });
        vector<RotatedRect> selected;
        for (const auto &c:candidates) {
            selected.push_back(rects[c.second]);
        }
        rects.clear();
        rects.assign(selected.begin(), selected.end());
    }

}

void FireRuneSplitter::shrinkImg(const GpuMat &src, cv::cuda::GpuMat &dst, float ratio_x, float ratio_y) {
    float w = src.cols * (1 - ratio_x);
    float h = src.rows * (1 - ratio_y);
    float x1 = (src.cols - w) / 2.0;
    float y1 = (src.rows - h) / 2.0;
    Rect window(x1, y1, w, h);
    dst = src(window);
}


