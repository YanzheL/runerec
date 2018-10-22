//
// Created by Trinity on 19/10/2018.
//

#ifndef RUNEREC_CLASSIFIER_INTERNAL_H
#define RUNEREC_CLASSIFIER_INTERNAL_H

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/types_c.h>
#include "opencv2/imgproc.hpp"

namespace runerec {
class ModelConfig {
  std::string modelDir;
};

class DigitClassifier {
 public:
  explicit DigitClassifier(int pfsize, const std::string &modelDir) : preferSize(pfsize), modelDir(modelDir) {}

  virtual void recognize(const std::vector<cv::Mat> &images, int *dst) = 0;

  virtual void recognize(const std::vector<cv::Mat> &images, int goal, int *pos) = 0;

//    virtual void init() = 0;

  inline void setPreferSize(int size) noexcept {
    this->preferSize = size;
  }

  inline int getPreferSize() const noexcept {
    return preferSize;
  }

  inline void regularize(const cv::Mat &img, cv::Mat &cropped) const {
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
  const std::string &modelDir;
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
}

#endif //RUNEREC_CLASSIFIER_INTERNAL_H
