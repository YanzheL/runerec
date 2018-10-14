//
// Created by LI YANZHE on 18-4-13.
//

#include <thread>
#include "Classifiers/Classifier.h"
#include "Splitters/RuneSplitter.h"

#define TEST_SPLITTER
//#define TEST_DETECTOR
//#define TEST_LED
//#define TEST_CAR
using namespace cv;
using namespace std;

#ifdef TEST_SPLITTER

#include "Splitters/RuneSplitter.h"
#include "Splitters/FireRuneSplitter.h"

int main() {
    Mat frame;
    VideoCapture capture("/home/nvidia/CLionProjects/runerecQt/1_cut.avi");
    if (!capture.isOpened())
        cout << "fail to open!" << endl;
//    RuneDetector detector;

    while (capture.read(frame)) {
//        imshow("frame",frame);
//        resize(frame,frame,Size(640,480));
        RuneSplitter *splitter = new FireRuneSplitter(130, 60);
//        RuneSplitter* splitter=new PureRuneSplitter(130,60);
        vector<Mat> blocks;
        double startTime = (double) cvGetTickCount();
        std::vector<cv::RotatedRect> sudoku_rects;
        splitter->split(frame, blocks, sudoku_rects);
        double endTime = (double) cvGetTickCount();
        cout << "Time = " << (endTime - startTime) / (cvGetTickFrequency() * 1000) << endl;
        for (int i = 0; i < blocks.size(); ++i) {
            auto block = blocks[i];
            imshow(to_string(i), block);
        }
        delete splitter;
        if (waitKey(1) == 27)
            break;
    }
}

#endif

#ifdef TEST_DETECTOR

//#include "RuneSplitter.h"
#include "RuneDetector.hpp"

int main() {
    Mat frame;
//    adjustExposure();
    VideoCapture capture("/home/nvidia/CLionProjects/runerecQt/small.avi");
//    capture.set(CV_CAP_PROP_EXPOSURE,164);
    if (!capture.isOpened()) {
        cout << "fail to open!" << endl;
        exit(1);
    }

    RuneDetector detector;

    while (capture.read(frame)) {
//        Rect r(Point2f(200, 0), Point2f(1640, 1080));
//        frame = frame(r);
//        resize(frame, frame, Size(640, 480));
        std::vector<cv::RotatedRect> sudoku_rects;
        int goal = 1;
        double startTime = (double) cvGetTickCount();
        int pos = detector.getTarget(frame, &goal, RuneDetector::RUNE_SMALL);
        double endTime = (double) cvGetTickCount();
        if (pos != -1) {
            rectangle(frame, detector.getRect(pos).boundingRect(), Scalar(0, 255, 0), 3);
            putText(frame, "Pos = " + to_string(pos), Point(100, 100), FONT_HERSHEY_PLAIN, 5.0, CV_RGB(0, 255, 0), 2);
        }
        cout << "Time = " << (endTime - startTime) / (cvGetTickFrequency() * 1000) << endl;
        imshow("frame", frame);
        if (waitKey(100) == 27)
            break;
    }
}

#endif

#ifdef TEST_LED


int main() {
    Mat frame;
    VideoCapture capture("/home/nvidia/CLionProjects/runerecQt/1.avi");
    if (!capture.isOpened())
        cout << "fail to open!" << endl;
//    RuneDetector detector;

    while (capture.read(frame)) {
//        imshow("frame",frame);
//        resize(frame,frame,Size(640,480));
        RuneSplitter* splitter=new LedRuneSplitter(20,35);
//        RuneSplitter* splitter=new PureRuneSplitter(130,60);
        vector<Mat> blocks;
        double startTime = (double) cvGetTickCount();
        std::vector<cv::RotatedRect> sudoku_rects;
        vector<std::vector<cv::Point2i>> contours;
//        cuda::GpuMat src(frame);
//        splitter->getContours(src,src,contours);
        splitter->split(frame,blocks, sudoku_rects);
        double endTime = (double) cvGetTickCount();
        cout<<"Block size = "<<blocks.size()<<endl;
        cout << "Time = " << (endTime - startTime) / (cvGetTickFrequency() * 1000) << endl;
        for (int i = 0; i < blocks.size(); ++i) {
            auto block = blocks[i];
            imshow(to_string(i), block);
        }
        delete splitter;
        if (waitKey(1) == 27)
            break;
    }
}

#endif
