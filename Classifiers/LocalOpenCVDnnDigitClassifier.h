//
// Created by LI YANZHE on 18-4-13.
//

#ifndef HANDWRITING_FLAMEWORDS_MODULE_2_LOCALOPENCVDNNDIGITCLASSIFIER_H
#define HANDWRITING_FLAMEWORDS_MODULE_2_LOCALOPENCVDNNDIGITCLASSIFIER_H

#include "Classifier_Internal.h"

class LocalOpenCVDnnDigitClassifier : public DigitClassifier {
public:

    LocalOpenCVDnnDigitClassifier() {
        preferSize = 28;
    }

    virtual long getId() {
        return uuid;
    }

    virtual void recognize(const std::vector<cv::Mat> &images, int *dst);

    virtual void recognize(const std::vector<cv::Mat> &images, int goal, int *pos);

protected:
    cv::dnn::Net net;

    virtual void recognize(cv::Mat &blob, int *dst, size_t batchsize);

    long uuid;
};

#endif //HANDWRITING_FLAMEWORDS_MODULE_2_LOCALOPENCVDNNDIGITCLASSIFIER_H
