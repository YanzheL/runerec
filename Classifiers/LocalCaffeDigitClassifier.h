//
// Created by LI YANZHE on 18-4-13.
//

#ifndef HANDWRITING_FLAMEWORDS_MODULE_2_LOCALCAFFEDIGITCLASSIFIER_H
#define HANDWRITING_FLAMEWORDS_MODULE_2_LOCALCAFFEDIGITCLASSIFIER_H

#include "LocalOpenCVDnnDigitClassifier.h"

class LocalCaffeDigitClassifier : public LocalOpenCVDnnDigitClassifier {
public:
    explicit LocalCaffeDigitClassifier(const string modelDir) : modelDir(modelDir) {
        uuid = BKDRHash(className + modelDir);
    };

    virtual void init() {
        string config = modelDir + "/config.prototxt";
        string model = modelDir + "/model.caffemodel";
        net = dnn::readNetFromCaffe(config, model);
    }

    virtual const string &getName() {
        return className;
    }

protected:
    const string modelDir;


private:
    const string className = "LocalCaffeDigitClassifier";
};

#endif //HANDWRITING_FLAMEWORDS_MODULE_2_LOCALCAFFEDIGITCLASSIFIER_H
