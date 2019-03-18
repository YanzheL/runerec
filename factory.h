//
// Created by Trinity on 2018-10-21.
//

#ifndef RUNEREC_FACTORY_H
#define RUNEREC_FACTORY_H

#include <string>
#include <iostream>
#include <memory>
#include <any>
#include <tuple>
#include <mutex>
#include <unordered_map>

namespace runerec {
template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
for_each_in_tuple(std::tuple<Tp...> &, FuncT) {}

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type
for_each_in_tuple(std::tuple<Tp...> &t, FuncT f) {
  f(std::get<I>(t));
  for_each_in_tuple<I + 1, FuncT, Tp...>(t, f);
}

class CachedFactory {
 public:
  template<class T, typename ...Arg>
  static std::shared_ptr<T> getInstance(Arg &&... param1) {
    auto &&params_tp = std::forward_as_tuple(std::forward<Arg>(param1)...);
    unsigned long param_hash = typeid(T).hash_code();
    unsigned long *pph = &param_hash;
    for_each_in_tuple(
        params_tp,
        [pph](auto x) {
          *pph += std::hash<decltype(x)>()(x);
        }
    );
    return getInstance<T, std::string>(param_hash, std::forward<Arg>(param1)...);
  }

 private:
  static std::mutex m;
  static std::unordered_map<unsigned long, std::any> instances;

  template<class T, typename ...Arg>
  static std::shared_ptr<T> getInstance(unsigned long id, Arg &&... param1) {
    std::lock_guard<std::mutex> guard(m);
    auto itr = instances.find(id);
    if (itr != instances.end()) {
      return std::any_cast<std::shared_ptr<T>>(itr->second);
    } else {
      auto t_start = std::chrono::high_resolution_clock::now();
      std::shared_ptr<T> ins(new T(std::forward<Arg>(param1)...));
      auto t_end = std::chrono::high_resolution_clock::now();
      float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
      std::cout << "Class<" << typeid(T).name() << "> Init success, hash = " << id << " time used = " << ms << " ms"
                << std::endl;
      instances[id] = std::any(ins);
      return ins;
    }
  }
};
}
#endif //RUNEREC_FACTORY_H
