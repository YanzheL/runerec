//
// Created by LI YANZHE on 18-4-13.
//

#ifndef HANDWRITING_FLAMEWORDS_MODULE_2_LOCALCAFFEDIGITCLASSIFIER_H
#define HANDWRITING_FLAMEWORDS_MODULE_2_LOCALCAFFEDIGITCLASSIFIER_H

#include "LocalOpenCVDnnDigitClassifier.h"

class LocalCaffeDigitClassifier : public LocalOpenCVDnnDigitClassifier {
public:
    explicit LocalCaffeDigitClassifier(const std::string modelDir) : modelDir(modelDir) {
        uuid = BKDRHash(className + modelDir);
    };

    virtual void init() {
        std::string config = modelDir + "/config.prototxt";
        std::string model = modelDir + "/model.caffemodel";
        net = cv::dnn::readNetFromCaffe(config, model);
    }

    virtual const std::string &getName() {
        return className;
    }

protected:
    const std::string modelDir;


private:
    const std::string className = "LocalCaffeDigitClassifier";
};

#endif //HANDWRITING_FLAMEWORDS_MODULE_2_LOCALCAFFEDIGITCLASSIFIER_H
