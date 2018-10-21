//
// Created by LI YANZHE on 18-4-13.
//

#ifndef RUNEREC_LOCALCAFFEDIGITCLASSIFIER_H
#define RUNEREC_LOCALCAFFEDIGITCLASSIFIER_H

#include "LocalOpenCVDnnDigitClassifier.h"

namespace runerec {
class LocalCaffeDigitClassifier : public LocalOpenCVDnnDigitClassifier {
 public:
  explicit LocalCaffeDigitClassifier(const std::string &modelDir) : LocalOpenCVDnnDigitClassifier(modelDir) {
    std::string config = modelDir + "/config.prototxt";
    std::string model = modelDir + "/model.caffemodel";
    net = cv::dnn::readNetFromCaffe(config, model);
  };
};
}

#endif //RUNEREC_LOCALCAFFEDIGITCLASSIFIER_H
