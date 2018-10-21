//
// Created by LI YANZHE on 18-4-13.
//

#include "LocalTensorRTDigitClassifier.h"

using namespace runerec;
using namespace std;
using namespace cv;
using namespace nvuffparser;
using namespace nvinfer1;

LocalTensorRTDigitClassifier::LocalTensorRTDigitClassifier(const string &modelDir) : modelDir(modelDir) {
  preferSize = 28;
  int maxBatchSize = 9;
  auto parser = createUffParser();
  parser->registerInput("Placeholder", DimsCHW(1, preferSize, preferSize));
  parser->registerOutput("Softmax");
  engine = loadModelAndCreateEngine(modelDir.c_str(), maxBatchSize, parser);
  parser->destroy();
  context = engine->createExecutionContext();
};

LocalTensorRTDigitClassifier::~LocalTensorRTDigitClassifier() {
  context->destroy();
  engine->destroy();
  nvuffparser::shutdownProtobufLibrary();
};

void *LocalTensorRTDigitClassifier::safeCudaMalloc(size_t memSize) {
  void *deviceMem;
  CHECK(cudaMalloc(&deviceMem, memSize));
  if (deviceMem == nullptr) {
    std::cerr << "LocalTensorRTDigitClassifier::safeCudaMalloc(" << memSize << ") Out of memory" << endl;
    exit(1);
  }
  return deviceMem;
}

std::vector<std::pair<int64_t, nvinfer1::DataType>>
LocalTensorRTDigitClassifier::calculateBindingBufferSizes(int nbBindings, int batchSize) {
  std::vector<std::pair<int64_t, nvinfer1::DataType>> sizes;
  for (int i = 0; i < nbBindings; ++i) {
    Dims dims = engine->getBindingDimensions(i);
    nvinfer1::DataType dtype = engine->getBindingDataType(i);
    int64_t eltCount = volume(dims) * batchSize;
    sizes.push_back(std::make_pair(eltCount, dtype));
  }
  return sizes;
}

void *LocalTensorRTDigitClassifier::createCudaBatch(nvinfer1::DataType dtype, const vector<Mat> &images) {
  int batchSize = images.size();
  /* in that specific case, eltCount == INPUT_H * INPUT_W */
  int64_t eltCount = batchSize * preferSize * preferSize;
  assert(elementSize(dtype) == sizeof(float));
  size_t memSize = eltCount * elementSize(dtype);
  float *inputs = new float[eltCount];
  /* initialize the inputs buffer */
  for (int i = 0; i < batchSize; ++i) {
    const Mat &img = images[i];
    Mat regularized;
    regularize(img, regularized);
    int sampleStart = i * preferSize * preferSize;
    for (int r = 0; r < preferSize; ++r)
      for (int c = 0; c < preferSize; ++c)
        inputs[sampleStart + r * preferSize + c] = regularized.at<uchar>(r, c);
  }
  void *deviceMem = safeCudaMalloc(memSize);
  CHECK(cudaMemcpy(deviceMem, inputs, memSize, cudaMemcpyHostToDevice));
  delete[] inputs;
  return deviceMem;
}

void LocalTensorRTDigitClassifier::recognize(const vector<Mat> &images, int *dst) {
//    for (auto img : images){
//        imshow("img in batch",img);
//        waitKey(0);
//    }
  int batchSize = images.size();
  if (batchSize <= 0)
    return;
  cout << "Classifier batch size = " << batchSize << endl;
  int nbBindings = engine->getNbBindings();
  assert(nbBindings == 2);
  std::vector<void *> buffers(nbBindings);
  auto buffersSizes = calculateBindingBufferSizes(nbBindings, batchSize);
  int bindingIdxInput = 0;
  for (int i = 0; i < nbBindings; ++i) {
    if (engine->bindingIsInput(i))
      bindingIdxInput = i;
    else {
      auto bufferSizesOutput = buffersSizes[i];
      buffers[i] = safeCudaMalloc(bufferSizesOutput.first * elementSize(bufferSizesOutput.second));
    }
  }
  auto bufferSizesInput = buffersSizes[bindingIdxInput];
  buffers[bindingIdxInput] = createCudaBatch(bufferSizesInput.second, images);

//    auto t_start = std::chrono::high_resolution_clock::now();
  context->execute(batchSize, &buffers[0]);
//    auto t_end = std::chrono::high_resolution_clock::now();
//    ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
//    total += ms;
  for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx) {
    if (engine->bindingIsInput(bindingIdx))
      continue;
    auto bufferSizesOutput = buffersSizes[bindingIdx];
    int64_t eltCount = bufferSizesOutput.first;
    nvinfer1::DataType dtype = bufferSizesOutput.second;
    size_t memSize = eltCount * elementSize(dtype);
    float *outputs = new float[eltCount];
    CHECK(cudaMemcpy(outputs, buffers[bindingIdx], memSize, cudaMemcpyDeviceToHost));
    for (int i = 0; i < batchSize; ++i) {
      int maxIdx = 0;
      float maxProb = 0.0;
      int s = i * 10;
      for (int j = 0; j < 10; ++j) {
        if (outputs[s + j] > outputs[maxIdx]) {
          maxIdx = s + j;
          maxProb = outputs[s];
        }
      }
      dst[i] = maxProb > 0 ? maxIdx - s : -1;
    }
    delete[] outputs;
  }
  CHECK(cudaFree(buffers[bindingIdxInput]));
  for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
    if (!engine->bindingIsInput(bindingIdx))
      CHECK(cudaFree(buffers[bindingIdx]));
  cout << "Classifier result = ";
  for (int i = 0; i < batchSize; ++i) {
    cout << *(dst + i) << ", ";
  }
  cout << endl;
}

ICudaEngine *
LocalTensorRTDigitClassifier::loadModelAndCreateEngine(const char *uffFile, int maxBatchSize, IUffParser *parser) {
  IBuilder *builder = createInferBuilder(gLogger);
  INetworkDefinition *network = builder->createNetwork();
  if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
    RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
  /* we create the engine */
  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(MAX_WORKSPACE);
  ICudaEngine *engine = builder->buildCudaEngine(*network);
  if (!engine)
    RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");
  /* we can clean the network and the parser */
  network->destroy();
  builder->destroy();
  return engine;
}

void LocalTensorRTDigitClassifier::recognize(const vector<Mat> &images, int goal, int *pos) {
  if (goal == -1) {
    *pos = -1;
    return;
  }
  int batchSize = images.size();
  auto result = new int[batchSize];
  recognize(images, result);
  *pos = -1;
  for (int i = 0; i < batchSize; ++i) {
    if (result[i] == goal) {
      *pos = i;
      break;
    }
  }
  delete[] result;
}