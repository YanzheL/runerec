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

using namespace cv;
using namespace std;

class FireDigitTest : public ::testing::Test {

protected:

    static void SetUpTestCase() {
        clsf = DigitClassifier::getInstance<LocalTFDigitClassifier>("LocalTFDigitClassifier",
                                                                    "../../models/mnist/model.pb");
    }

    static void TearDownTestCase() {

    }

    static DigitClassifier *clsf;

};

DigitClassifier *FireDigitTest::clsf;

namespace fs = boost::filesystem;

TEST_F(FireDigitTest, F_Img) {
    string dir = "../../f_num";
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
    FireDigitTest::clsf->recognize(imgs, res);
    EXPECT_THAT(answers, ::testing::ElementsAreArray(res));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}