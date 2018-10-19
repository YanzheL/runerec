//
// Created by LI YANZHE on 18-1-30.
//

#include "Classifier.h"

using namespace std;
using namespace cv;
map<int, DigitClassifier *> DigitClassifier::instances;

#ifdef USE_REMOTE

void HWDigitClassifier::add_item(long seq, int goal, int pos, const Mat &img, const RotatedRect &rotRect,
                            RepeatedData *container) {
//    Mat M, rotated, cropped;
//    Mat* M=Arena::Create<Mat>(arena);
    auto rotated = Arena::Create<Mat>(arena);
    auto cropped = Arena::Create<Mat>(arena);
    float angle = rotRect.angle;
    Size rect_size = rotRect.size;
    if (rotRect.angle < -45.) {
        angle += 90.0;
        int tp = rect_size.width;
        rect_size.width = rect_size.height;
        rect_size.height = tp;
    }
    Mat M = getRotationMatrix2D(rotRect.center, angle, 1.0);
    // perform the affine transformation
    warpAffine(img, *rotated, M, img.size());
    // crop the resulting image
    getRectSubPix(*rotated, rect_size, rotRect.center, *cropped);
    add_item(seq, goal, pos, *cropped, container);
}

void HWDigitClassifier::add_item(long seq, int goal, int pos, const Mat &img, RepeatedData *container) {
    auto seqPack = Arena::CreateMessage<UInt64Value>(arena);
    auto goalPack = Arena::CreateMessage<UInt32Value>(arena);
    auto posPack = Arena::CreateMessage<Int32Value>(arena);
    auto cell = Arena::CreateMessage<Cell>(arena);

    seqPack->set_value(seq);
    goalPack->set_value(goal);
    posPack->set_value(pos);
    cell->set_allocated_goal(goalPack);
    cell->set_allocated_seq(seqPack);
    cell->set_allocated_pos(posPack);
    Mat dataImg(img);

    if (img.channels() != 1)
        cvtColor(img, dataImg, CV_BGR2GRAY);
    int preferred_height = (int) (((double) preferSize / (double) dataImg.cols) * dataImg.rows);
    resize(dataImg, dataImg, Size(preferSize, preferred_height));
//    imshow("a",dataImg);
//    waitKey(0);
    auto imgMsg = toImage(dataImg);
#ifdef SHOW_BINARY_MAT
    cout << endl;
    for (int i = 0; i < dataImg.rows * dataImg.cols; ++i) {
        if (i % dataImg.cols == 0 && i > 0) cout << endl;
        printf("%4d", *(dataImg.data + i));
    }
    cout << endl;
#endif
    cell->set_allocated_img(imgMsg);
    container->add_items()->PackFrom(*cell);
}

void HWDigitClassifier::recognize(RepeatedData *container, int *dst) {
    ++callTimes;
    auto result = Arena::CreateMessage<Cell>(arena);
    try {
#ifdef DEBUG_COMMUNICATOR_V1
        communicator->send(*container, method).UnpackTo(result);
        printf("Reply:\nSeq = %ld, Goal = %d, Pos = %d\n",
               result->seq().value(),
               result->goal().value(),
               result->pos().value());
#endif
    }
    catch (std::exception &e) {
        std::cerr << "Exception: " << e.what() << endl;
    }
    dst[0] = result->seq().value();
    dst[1] = result->goal().value();
    dst[2] = result->pos().value();
    if (callTimes % 100 == 0) {
#ifdef DEBUG_ARENA
        printf("---------- Before: Space allocated = %ld, used = %ld ----------\n", arena->SpaceAllocated(),
               arena->SpaceUsed());
#endif
        arena->Reset();
#ifdef DEBUG_ARENA
        printf("---------- After:  Space allocated = %ld, used = %ld ----------\n", arena->SpaceAllocated(),
               arena->SpaceUsed());
#endif
    }
}


void LedDigitClassifier::recognize(vector<Mat> &images, int *dst) {
    RepeatedData *container = allocBatchContainer();
    vector<Mat>::iterator it;
    for (it = images.begin(); it != images.end(); ++it) {
        Any *item = container->add_items();
        Mat cropped;
        if (it->channels() != 1)
            cvtColor(*it, cropped, CV_BGR2GRAY);
        resize(cropped, cropped, Size(preferSize, preferSize));
        item->PackFrom(*toImage(cropped));
    }
    recognize(container, dst);
}

void LedDigitClassifier::recognize(RepeatedData *container, int *dst) {
    RepeatedData *targets = Arena::CreateMessage<RepeatedData>(arena);
    communicator->send(*container, RECO_LED_DIGIT).UnpackTo(targets);
    auto ptr = targets->items();
    RepeatedPtrField<const Any>::iterator targetItr;
    int i = 0;
    for (targetItr = ptr.begin(); targetItr != ptr.end(); ++targetItr, ++i) {
        UInt32Value val;
        targetItr->UnpackTo(&val);
        *(dst + i) = val.value();
    }
}
#endif

//#define DEBUG_NET_STRUCTURE

#ifdef USE_CUDA


#endif
