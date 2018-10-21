//
// Created by Trinity on 18/10/2018.
//

//
// Created by LI YANZHE on 18-4-13.
//


#include <string>
#include <boost/filesystem.hpp>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "../Classifiers/Classifier.h"
#include "../factory.h"

using namespace cv;
using namespace std;
using namespace runerec;

class AllModelFixture : public ::testing::Test {

 protected:

  static void SetUpTestCase() {
    tf_clsf = CachedFactory::getInstance<LocalTFDigitClassifier>("../../../models/mnist/model.pb");
    led_clsf = CachedFactory::getInstance<LocalTFDigitClassifier>("../../../models/led/model.pb");
    cf_clsf = CachedFactory::getInstance<LocalCaffeDigitClassifier>("../../../models/caffe");
  }

  static void TearDownTestCase() {

  }

  static std::shared_ptr<DigitClassifier> tf_clsf, cf_clsf, led_clsf;

  double accuracy(int res[], const vector<int> &answer) {
    double t = 0;
    double s = answer.size();
    for (int i = 0; i < s; ++i) {
      if (res[i] == answer[i])
        ++t;
    }
    return t / s;
  }

};

std::shared_ptr<DigitClassifier>AllModelFixture::tf_clsf;
std::shared_ptr<DigitClassifier>AllModelFixture::cf_clsf;
std::shared_ptr<DigitClassifier>AllModelFixture::led_clsf;

namespace fs = boost::filesystem;

TEST_F(AllModelFixture, F_Img) {
  string dir = "../../tests/data/fire_digits";
  vector<Mat> imgs;
  vector<int> answers;
  int res[9];
  for (auto &p : fs::directory_iterator(dir)) {
    string path = p.path().string();
    int answer = *(path.rbegin() + 4) - '0';
    Mat frame;
    frame = imread(path, 0);
    threshold(frame, frame, 127, 1, THRESH_BINARY_INV);
    imgs.push_back(frame);
    answers.push_back(answer);
  }
  AllModelFixture::tf_clsf->recognize(imgs, res);
  EXPECT_THAT(answers, ::testing::ElementsAreArray(res));
}

TEST_F(AllModelFixture, F_Img_Caffe) {
  string dir = "../../tests/data/fire_digits";
  vector<Mat> imgs;
  vector<int> answers;
  int res[9];
  for (auto &p : fs::directory_iterator(dir)) {
    string path = p.path().string();
    int answer = *(path.rbegin() + 4) - '0';
    Mat frame;
    frame = imread(path, 0);
    threshold(frame, frame, 127, 1, THRESH_BINARY_INV);
    imgs.push_back(frame);
    answers.push_back(answer);
  }
  AllModelFixture::cf_clsf->recognize(imgs, res);
  EXPECT_GE(accuracy(res, answers), 0.5);
//  EXPECT_THAT(answers, ::testing::ElementsAreArray(res));
}

TEST_F(AllModelFixture, LED_Img) {
  string dir = "../../tests/data/led_digits";
  vector<Mat> imgs;
  vector<int> answers;
  int res[9];
  for (auto &p : fs::directory_iterator(dir)) {
    string path = p.path().string();
    int answer = *(path.rbegin() + 4) - '0';
    Mat frame;
    frame = imread(path, 0);
    threshold(frame, frame, 127, 1, THRESH_BINARY);
    imgs.push_back(frame);
    answers.push_back(answer);
  }
  AllModelFixture::led_clsf->recognize(imgs, res);
  EXPECT_THAT(answers, ::testing::ElementsAreArray(res));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}