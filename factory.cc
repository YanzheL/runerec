//
// Created by Trinity on 2018-10-21.
//

#include "factory.h"

using namespace runerec;

std::unordered_map<unsigned long, std::any> CachedFactory::instances;
std::mutex CachedFactory::m;