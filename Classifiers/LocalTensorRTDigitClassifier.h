//
// Created by LI YANZHE on 18-4-13.
//

#ifndef HANDWRITING_FLAMEWORDS_MODULE_2_LOCALTENSORRTDIGITCLASSIFIER_H
#define HANDWRITING_FLAMEWORDS_MODULE_2_LOCALTENSORRTDIGITCLASSIFIER_H

#include "Classifier_Internal.h"
#include <algorithm>
#include "cuda_runtime_api.h"
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"
#include "common.h"

static Logger gLogger;

#define MAX_WORKSPACE (1 << 30)

#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "sample_uff_mnist: " + std::string(message);            \
        gLogger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        return (ret);                                                                       \
    } while(0)

class LocalTensorRTDigitClassifier : public DigitClassifier {
public:
    explicit LocalTensorRTDigitClassifier(const string modelDir) : modelDir(modelDir) {
        uuid = BKDRHash(className + modelDir);
        preferSize = 28;
    };

    virtual long getId() {
        return uuid;
    }

    virtual void init();

    virtual const string &getName() {
        return className;
    }

    virtual ~LocalTensorRTDigitClassifier() {
        context->destroy();
        engine->destroy();
        nvuffparser::shutdownProtobufLibrary();
    }

    virtual void recognize(const std::vector<cv::Mat> &images, int *dst);

    virtual void recognize(const std::vector<cv::Mat> &images, int goal, int *pos);

protected:
    const string modelDir;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;

    inline static int64_t volume(const nvinfer1::Dims &d) {
        int64_t v = 1;
        for (int64_t i = 0; i < d.nbDims; i++)
            v *= d.d[i];
        return v;
    }

    inline static unsigned int elementSize(nvinfer1::DataType t) {
        switch (t) {
            case nvinfer1::DataType::kFLOAT:
                return 4;
            case nvinfer1::DataType::kHALF:
                return 2;
            case nvinfer1::DataType::kINT8:
                return 1;
        }
        return 0;
    }

    void *safeCudaMalloc(size_t memSize);

    std::vector<std::pair<int64_t, nvinfer1::DataType>>
    calculateBindingBufferSizes(int nbBindings, int batchSize);

    void *createCudaBatch(nvinfer1::DataType dtype, const std::vector<cv::Mat> &images);

    nvinfer1::ICudaEngine *loadModelAndCreateEngine(const char *uffFile, int maxBatchSize,
                                                    nvuffparser::IUffParser *parser);

private:
    const string className = "LocalTensorRTDigitClassifier";
    long uuid;
};

#endif //HANDWRITING_FLAMEWORDS_MODULE_2_LOCALTENSORRTDIGITCLASSIFIER_H
