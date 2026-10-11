#pragma once
// Metadata-only counters and measured intervals. No tensor values are read.
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <utility>

namespace execution_observe {
inline bool force_generic() {
    const char *v = std::getenv("OURS_VIT_FORCE_GENERIC_ALL_S");
    return v && std::string(v) == "1";
}
inline std::array<std::atomic<unsigned long long>, 4> he_counts{};
template <class Function> decltype(auto) he_call(int kind, Function &&f) {
    he_counts[kind].fetch_add(1, std::memory_order_relaxed);
    return std::forward<Function>(f)();
}
inline std::array<unsigned long long, 4> snapshot() {
    return {he_counts[0].load(), he_counts[1].load(),
            he_counts[2].load(), he_counts[3].load()};
}
inline void interval(const char *kind, const char *function, int line,
                     std::chrono::steady_clock::time_point begin) {
    const auto end = std::chrono::steady_clock::now();
    std::cout << "EXECUTION_INTERVAL kind=" << kind << " function=" << function
              << " source_line=" << line << " begin_ns="
              << std::chrono::duration_cast<std::chrono::nanoseconds>(begin.time_since_epoch()).count()
              << " end_ns="
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count()
              << " payload_values_logged=0" << std::endl;
}
struct Region {
    const char *kind, *function;
    int line;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    Region(const char *k, const char *f, int l):kind(k),function(f),line(l){}
    ~Region(){interval(kind, function, line, begin);}
};
}
