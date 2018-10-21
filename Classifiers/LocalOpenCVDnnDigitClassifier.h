//
// Created by LI YANZHE on 18-4-13.
//

#ifndef RUNEREC_LOCALOPENCVDNNDIGITCLASSIFIER_H
#define RUNEREC_LOCALOPENCVDNNDIGITCLASSIFIER_H

#include "Classifier_Internal.h"
#include "opencv2/dnn.hpp"

namespace runerec {
class LocalOpenCVDnnDigitClassifier : public DigitClassifier {
 public:

  LocalOpenCVDnnDigitClassifier(const std::string &modelDir) : DigitClassifier(28, modelDir) {}

  void recognize(const std::vector<cv::Mat> &images, int *dst) override;

  void recognize(const std::vector<cv::Mat> &images, int goal, int *pos) override;

 protected:

  cv::dnn::Net net;

  virtual void recognize(cv::Mat &blob, int *dst, size_t batchsize);
};
}

#endif //RUNEREC_LOCALOPENCVDNNDIGITCLASSIFIER_H
