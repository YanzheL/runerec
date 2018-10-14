//
// Created by LI YANZHE on 18-4-13.
//

#ifndef HANDWRITING_FLAMEWORDS_MODULE_2_LOCALTFDIGITCLASSIFIER_H
#define HANDWRITING_FLAMEWORDS_MODULE_2_LOCALTFDIGITCLASSIFIER_H

#include "LocalOpenCVDnnDigitClassifier.h"

class LocalTFDigitClassifier : public LocalOpenCVDnnDigitClassifier {
public:
    explicit LocalTFDigitClassifier(const std::string modelDir) : modelDir(modelDir) {
        uuid = BKDRHash(className + modelDir);
    };

    virtual void init() {
        std::string model = modelDir;
        net = cv::dnn::readNetFromTensorflow(model);
#ifdef DEBUG_NET_STRUCTURE
        vector<std::string> names=net.getLayerNames();
    for(const std::string& n:names)
        cout<<n<<endl;
#endif
    }

    virtual const std::string &getName() {
        return className;
    }

protected:
    const std::string modelDir;

private:
    const std::string className = "LocalTFDigitClassifier";
};

#endif //HANDWRITING_FLAMEWORDS_MODULE_2_LOCALTFDIGITCLASSIFIER_H
