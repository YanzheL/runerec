//
// Created by LI YANZHE on 18-4-13.
//

//#define TEST_SPLITTER
#define TEST_DETECTOR
//#define TEST_LED
//#define TEST_CAR


#ifdef TEST_SPLITTER

#include "Splitters/RuneSplitter.h"
#include "Splitters/FireRuneSplitter.h"
#include "Splitters/PureRuneSplitter.h"

int main() {
    Mat frame;
    VideoCapture capture("/home/trinity/CLionProjects/1.avi");
    if (!capture.isOpened())
        cout << "fail to open!" << endl;
//    RuneDetector detector;

    while (capture.read(frame)) {
//        imshow("frame",frame);
//        resize(frame,frame,Size(640,480));
//        RuneSplitter *splitter = new FireRuneSplitter(130, 60);
        RuneSplitter* splitter=new PureRuneSplitter(130,60);
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

#include <thread>
#include "Classifiers/Classifier.h"
#include <boost/filesystem.hpp>
#include "factory.h"
#include "Utils.h"

using namespace cv;
using namespace std;
using namespace runerec;
namespace fs = boost::filesystem;

int main() {
  std::shared_ptr<DigitClassifier> clsf, tf_clsf, ledClsf;
#ifdef USE_CUDA
//  ledClsf = CachedFactory::getInstance<LocalTensorRTDigitClassifier>("../../models/led/model.uff");
  clsf = CachedFactory::getInstance<LocalTensorRTDigitClassifier>("../../models/mnist/model.uff");
//    tf_clsf = DigitClassifier::getInstance<LocalTFDigitClassifier>("LocalTFDigitClassifier",
//                                                                "../../models/mnist/model.pb");
//    fireClsf = DigitClassifier::getInstance<LocalTensorRTDigitClassifier>("LocalTensorRTDigitClassifier",
//                                                                          "../../models/fire/model.uff");
#else
  clsf = CachedFactory::getInstance<LocalTFDigitClassifier>("../../models/mnist/model.pb");
#endif

  string dir = "../tests/data/fire_digits";
  vector<Mat> imgs;
  vector<int> answers;
  int res[9];
  for (auto &p : fs::directory_iterator(dir)) {
    string path = p.path().string();
    int answer = *(path.rbegin() + 4) - '0';
    Mat frame;
    frame = imread(path, 0);
    threshold(frame, frame, 127, 1, THRESH_BINARY_INV);
    imgs.push_back(frame);
    answers.push_back(answer);
  }

  while (true) {
    double startTime = (double) cvGetTickCount();
    clsf->recognize(imgs, res);
    double endTime = (double) cvGetTickCount();
    cout << "accuracy = " << accuracy(res, answers) << endl;
    cout << "Time = " << (endTime - startTime) / (cvGetTickFrequency() * 1000) << endl;
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
