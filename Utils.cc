//
// Created by LI YANZHE on 18-4-13.
//

#include "Utils.h"

using namespace std;

unsigned int BKDRHash(string src) {
    char *str = (char *) src.c_str();
    unsigned int seed = 131; // 31 131 1313 13131 131313 etc..
    unsigned int hash = 0;
    while (*str) {
        hash = hash * seed + (*str++);
    }
    return (hash & 0x7FFFFFFF);
}