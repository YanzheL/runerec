//
// Created by LI YANZHE on 18-4-13.
//

#ifndef RUNEREC_LOCALTFDIGITCLASSIFIER_H
#define RUNEREC_LOCALTFDIGITCLASSIFIER_H

#include "LocalOpenCVDnnDigitClassifier.h"

namespace runerec {
class LocalTFDigitClassifier : public LocalOpenCVDnnDigitClassifier {
 public:

  explicit LocalTFDigitClassifier(const std::string &modelDir) : modelDir(modelDir) {
    net = cv::dnn::readNetFromTensorflow(modelDir);
#ifdef DEBUG_NET_STRUCTURE
    vector<std::string> names=net.getLayerNames();
    for(const std::string& n:names)
        cout<<n<<endl;
#endif
  };

 protected:

  const std::string modelDir;
};
}

#endif //RUNEREC_LOCALTFDIGITCLASSIFIER_H
