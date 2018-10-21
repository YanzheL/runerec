//
// Created by Trinity on 2018-10-21.
//

#include "factory.h"

using namespace runerec;

//
//template<class T, typename ...Arg>
//std::shared_ptr<T> CachedFactory::getInstance(Arg &&... param1) {
//    auto &&params_tp = std::forward_as_tuple(std::forward<Arg>(param1)...);
////        long id = BKDRHash();
//    unsigned long param_hash = typeid(T).hash_code();
//    unsigned long *pph = &param_hash;
//    for_each_in_tuple(
//            params_tp,
//            [pph](auto x) {
//                *pph += std::hash<decltype(x)>()(x);
//            }
//    );
//    return getInstance<T, std::string>(param_hash, std::forward<Arg>(param1)...);
//}
//
//template<class T, typename ...Arg>
//std::shared_ptr<T> CachedFactory::getInstance(unsigned long id, Arg &&... param1) {
//    auto itr = instances.find(id);
//    if (itr != instances.end()) {
//        return std::any_cast<std::shared_ptr<T>>(itr->second);
//    } else {
//        auto t_start = std::chrono::high_resolution_clock::now();
//        std::shared_ptr<T> ins(new T(std::forward<Arg>(param1)...));
//        auto t_end = std::chrono::high_resolution_clock::now();
//        float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
//        std::cout << "Init success, time used = " << ms << " ms" << std::endl;
//        instances[id] = std::any(ins);
//        return ins;
//    }
//}

std::map<unsigned long, std::any> CachedFactory::instances;