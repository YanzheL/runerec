//
// Created by LI YANZHE on 18-4-13.
//

#include "LocalOpenCVDnnDigitClassifier.h"

using namespace std;
using namespace cv;

void LocalOpenCVDnnDigitClassifier::recognize(const vector<Mat> &images, int *dst) {
    Mat blob = dnn::blobFromImages(images, 1.0, Size(preferSize, preferSize), Scalar(0, 0, 0), false, false);
    recognize(blob, dst, images.size());
}

void LocalOpenCVDnnDigitClassifier::recognize(const vector<Mat> &images, int goal, int *pos) {
    if (goal == -1) {
        *pos = -1;
        return;
    }
    int i = 0;
    for (const Mat &image:images) {
        int answer;
        Mat blob = dnn::blobFromImage(image, 1.0, Size(preferSize, preferSize), Scalar(0, 0, 0), false, false);
        recognize(blob, &answer, 1);
        if (answer == goal) {
            *pos = i;
            return;
        }
        ++i;
    }
    *pos = -1;
}

void LocalOpenCVDnnDigitClassifier::recognize(Mat &blob, int *dst, size_t batchsize) {

    net.setInput(blob);
    Mat prob = net.forward();
    for (int i = 0; i < batchsize; ++i) {
        Mat row = prob.row(i);
        double min, max;
        int idx_min[2], idx_max[2];
        minMaxIdx(row, &min, &max, idx_min, idx_max);
#ifdef SHOW_MAX_PROB
        cout << "Max prob = " << max << endl;
#endif
        dst[i] = max > 0.7 ? idx_max[1] : -1;
    }
}