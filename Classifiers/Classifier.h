//
// Created by LI YANZHE on 18-1-30.
//

#ifndef RUNEREC_CLASSIFIER_H
#define RUNEREC_CLASSIFIER_H

#include "LocalOpenCVDnnDigitClassifier.h"

#ifdef USE_CUDA

#include "LocalTensorRTDigitClassifier.h"

#endif
#ifdef USE_CAFFE
#include "LocalCaffeDigitClassifier.h"
#endif

#include "LocalTFDigitClassifier.h"

#endif //RUNEREC_CLASSIFIER_H
