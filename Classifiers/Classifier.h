//
// Created by LI YANZHE on 18-1-30.
//

#ifndef RUNEREC_CLASSIFIER_H
#define RUNEREC_CLASSIFIER_H

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/dnn.hpp"

#include "../Utils.h"

//#define USE_CUDA
//#define USE_REMOTE

//#define DEBUG_NET_STRUCTURE
//#define IMSHOW_EACH_CELL
//#define SHOW_BINARY_MAT
//#define SHOW_MAX_PROB

//using namespace cv;
//using namespace std;


class DigitClassifier : public Identifiable {
public:
    template<class T, typename P>
    static T *getInstance(char *name, Identifiable *param1) {
        long id = BKDRHash(std::string(name)) + param1->getId();
        return getInstance<T, P>(id, param1);
    }

    template<class T>
    static T *getInstance(std::string name, std::string param1) {
        long id = BKDRHash(name + param1);
        return getInstance<T, std::string>(id, param1);
    }

    virtual void recognize(const std::vector<cv::Mat> &images, int *dst) = 0;

    virtual void recognize(const std::vector<cv::Mat> &images, int goal, int *pos) = 0;

    virtual void init() = 0;

    inline void setPreferSize(int size) {
        this->preferSize = size;
    }

    inline int getPreferSize() {
        return preferSize;
    }

    inline void regularize(const cv::Mat &img, cv::Mat &cropped) {
        if (!(img.rows == preferSize && img.cols == preferSize)) {
            resize(img, cropped, cv::Size(preferSize, preferSize));
        } else if (img.channels() != 1) {
            cvtColor(img, cropped, CV_BGR2GRAY);
        } else
            img.copyTo(cropped);
    }

    virtual ~DigitClassifier() {}

protected:
    int preferSize;
    static std::map<int, DigitClassifier *> instances;

    template<class T, typename P>
    static T *getInstance(long id, P param1) {
        auto itr = instances.find(id);
        if (itr != instances.end()) {
            return (T *) (itr->second);
        } else {
            auto ins = new T(param1);
            auto t_start = std::chrono::high_resolution_clock::now();
            ins->init();
            auto t_end = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
            std::cout << "Init success, time used = " << ms << " ms" << std::endl;
            instances[id] = ins;
            return ins;
        }
    }
};


#ifdef USE_REMOTE

//#define DEBUG_COMMUNICATOR
#define DEBUG_ARENA
#define DEBUG_COMMUNICATOR_V1
#include "proto/Image.pb.h"
#include "proto/RpcMessage.pb.h"
#include <boost/asio.hpp>
#include "Communicator.h"

using namespace google::protobuf;
using namespace org::yanzhe::robomaster::recodaemon::net::proto;
using boost::asio::ip::tcp;

class RemoteDigitClassifier : public DigitClassifier {

public:

    virtual void recognize(RepeatedData *container, int *dst)=0;

    inline void setMethod(RecoMethod method) {
        this->method = method;
    }

    inline RepeatedData *allocBatchContainer() {
        return Arena::CreateMessage<RepeatedData>(arena);
    }

protected:
    int preferSize;
    RecoMethod method;
    Arena *arena;
    Communicator *communicator;
    long callTimes;

    explicit RemoteDigitClassifier(Communicator *communicator) {
        this->communicator = communicator;
        this->arena = communicator->getArena();
    }

    Image *toImage(const cv::Mat &img) {
        auto imgMsg = Arena::CreateMessage<Image>(arena);
        auto rowsPack = Arena::CreateMessage<Int32Value>(arena);
        auto colsPack = Arena::CreateMessage<Int32Value>(arena);
        auto channelsPack = Arena::CreateMessage<Int32Value>(arena);
        rowsPack->set_value(img.rows);
        colsPack->set_value(img.cols);
        channelsPack->set_value(img.channels());
        imgMsg->set_allocated_rows(rowsPack);
        imgMsg->set_allocated_cols(colsPack);
        imgMsg->set_allocated_channels(channelsPack);
        size_t size = img.rows * img.cols * img.channels();
//        imshow("a",img);
//        waitKey(0);
        imgMsg->set_data(img.data, size);
        return imgMsg;

    }

    static std::map<int, RemoteDigitClassifier *> instances;
};

class LedDigitClassifier : public RemoteDigitClassifier {
public:

    void recognize(std::vector<cv::Mat> &images, int *dst);

    void recognize(RepeatedData *container, int *dst);

    void setMethod(RecoMethod method)= delete;

    long getId() {
        return uuid;
    }

    const std::string &getName() {
        return className;
    }

    explicit LedDigitClassifier(Communicator *communicator) : RemoteDigitClassifier(communicator) {
        method = RECO_LED_DIGIT;
        preferSize = 30;
        uuid = BKDRHash(std::string("LedDigitClassifier")) + communicator->getId();
    }

protected:


private:
    long uuid;

    const std::string className = std::string("LedDigitClassifier");
};


class HWDigitClassifier : public RemoteDigitClassifier {

public:

    void add_item(long seq, int goal, int pos, const cv::Mat &img, RepeatedData *container);

    void add_item(long seq, int goal, int pos, const cv::Mat &img, const RotatedRect &rotRect, RepeatedData *container);

    void recognize(RepeatedData *container, int *dst);

    long getId() {
        return uuid;
    }

    explicit HWDigitClassifier(Communicator *communicator) : RemoteDigitClassifier(communicator) {
        method = RECO_SIMPLE_HW_DIGIT;
        preferSize = 50;
        uuid = BKDRHash(std::string("HWDigitClassifier")) + communicator->getId();
    }

    const std::string &getName() {
        return className;
    }

protected:

private:
    long uuid;
    const std::string className = std::string("HWDigitClassifier");
};

#endif

#endif //RUNEREC_CLASSIFIER_H
